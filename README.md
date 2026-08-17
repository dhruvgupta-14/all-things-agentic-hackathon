# Research Paper Reading Companion

FastAPI + Google ADK agent over PostgreSQL/pgvector. See the architecture
document for the design; this file covers local setup only.

## Local setup

```bash
python -m venv venv
./venv/Scripts/python.exe -m pip install -r requirements.txt -r requirements-dev.txt

cp .env.example .env        # then fill in the values

docker compose up -d db     # Postgres 16 + pgvector on :5432
./venv/Scripts/python.exe -m alembic upgrade head
./venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

`alembic upgrade head` is the only supported way to create or change the
schema. It creates the `vector` extension, all 14 tables, and the append-only
triggers on `turns`, `observations`, and `quiz_attempts`.

## Database changes

Edit `app/db/models.py`, then:

```bash
./venv/Scripts/python.exe -m alembic revision --autogenerate -m "what changed"
./venv/Scripts/python.exe -m alembic upgrade head
```

Read the generated migration before applying it — autogenerate does not detect
trigger or extension changes, and those must be hand-written with `op.execute`.
`alembic check` fails when the models and the database have drifted apart.

## Embeddings and retrieval

With `VERTEX_PROJECT` unset, the system uses a deterministic **hashing
embedder** so ingestion and retrieval work offline. It is a hashing-trick
embedding, not random noise: texts sharing vocabulary genuinely score closer,
which makes retrieval testable locally. It is *lexical only* — it cannot match
"car" to "automobile" — so `RETRIEVAL_MIN_SIMILARITY` will need retuning once
`gemini-embedding-001` is in play. The model used is recorded on
`papers.embedding_model`, so a switch is detectable rather than silently
corrupting a mixed-vector index.

Set `VERTEX_PROJECT` to switch to real embeddings. Existing papers keep their
old vectors and must be re-ingested.

## Changing the embedding model

Vectors from two different models are not comparable — cosine distance between
them is a meaningless number that still sorts, so a mixed index returns
confident nonsense rather than an error. Three things prevent that:

* retrieval joins `papers` and serves only chunks whose `embedding_model`
  matches the active embedder, so a stale paper returns nothing rather than
  garbage;
* `GET /papers` reports `needs_reindex` per paper, so staleness is visible
  instead of looking like an empty result;
* `scripts/reindex.py` re-embeds the stale ones.

```bash
PYTHONPATH=. python scripts/reindex.py --list        # what is stale
PYTHONPATH=. python scripts/reindex.py --stale --dry-run
PYTHONPATH=. python scripts/reindex.py --stale       # re-embed them
PYTHONPATH=. python scripts/reindex.py --paper <uuid>
```

Idempotent: papers are committed one at a time, a failure rolls that paper back
untouched rather than degrading a working one, and a second `--stale` run finds
nothing. Re-indexing deliberately does not re-run per-reader concept
canonicalization — it is a vector operation, not a learner-model one.

## Cloud provisioning

```bash
./scripts/provision_gcp.sh --project <project> --dry-run   # review first
./scripts/provision_gcp.sh --project <project>
```

Idempotent and non-destructive: every step checks before creating. It sets up
the bucket (private, uniform access, public-access-prevention), the Cloud Tasks
queue, the service account and least-privilege IAM, then prints the staging
env block. It deliberately does not create Cloud SQL or download a
service-account key — see the notes it prints.

**When you create Cloud SQL, set the `random_page_cost` database flag to 1.1.**
The default of 4.0 is a spinning-disk figure and makes the planner answer
vector queries with a sequential scan instead of the HNSW index. Measured on a
5 000-chunk corpus: 183ms sequential vs 1ms indexed. The app also sets this
per-connection, so this is belt and braces.

## Retrieval benchmark

```bash
PYTHONPATH=. python scripts/benchmark_retrieval.py --papers 50 --chunks-per-paper 100
PYTHONPATH=. python scripts/benchmark_retrieval.py --cleanup
```

Seeds a synthetic corpus and reports whether each query shape uses HNSW, an
exact index scan, or a sequential scan. Run it after any change to retrieval
SQL or index definitions. The corpus is tagged so `--cleanup` removes exactly
what it created.

## Checks

```bash
./venv/Scripts/python.exe -m ruff check app scripts tests
./venv/Scripts/python.exe -m alembic check     # no model/schema drift
./venv/Scripts/python.exe -m pytest
```

`pytest` runs **offline**. It never calls Gemini, never reads `demo_papers/`,
and never assumes an empty database — every PDF it needs is generated in
`tests/conftest.py`, and the model backends fall back to deterministic stubs.

Test isolation is enforced structurally rather than by review. Each test
transaction is seeded with rows it does not own (`_seed_decoy_data`), so a test
that queries global state — `count(*) FROM papers`, "the only concept" — fails
immediately instead of passing on an empty database and breaking the first time
someone ingests a real paper. `tests/test_isolation.py` verifies that guard is
in force, and each test gets a unique auth subject rather than sharing
`local-dev-user` with a human developer.

## Conversation history

PostgreSQL owns the transcript, not ADK. Each turn hydrates a **throwaway
in-memory** ADK session from the `messages` table, runs the agent, and persists
the new `user` and `assistant` rows itself. ADK never writes to the database.

That buys three things: history survives an instance being reclaimed, the
transcript outlives any decision to change agent framework, and the reset-by-
replay demo script has real messages to replay.

`messages` is the single owner of conversation content — `turns` holds metadata
and provenance only. Rows are **immutable**: UPDATE is rejected by trigger,
because a transcript that can be rewritten is not a transcript. DELETE stays
permitted, because both the retention sweep and `ON DELETE CASCADE` from a
deleted user need it.

Tool calls and their results are deliberately **not** stored. They are working
memory for one turn, and their provenance already lives in `turn_retrievals`.

History older than **30 days** is deleted. The sweep runs on append, throttled
to once an hour per process, so it needs no scheduler and costs nothing while
the system is idle. For a long-idle deployment, or to see what would go:

```bash
PYTHONPATH=. python scripts/prune_messages.py --dry-run
PYTHONPATH=. python scripts/prune_messages.py
```

## Demo paper validation

Separate from the test suite, because it needs the real PDFs, a real Gemini
key with quota, and the persisted demo records:

```bash
PYTHONPATH=. python scripts/verify_demo_ingestion.py                    # check
PYTHONPATH=. python scripts/verify_demo_ingestion.py --ingest           # ingest first
PYTHONPATH=. python scripts/verify_demo_ingestion.py --rebuild-concepts # re-derive only
```

It verifies the ARCHITECTURE §12 step-0 precondition: after both papers are
ingested and **before any question is asked**, the concept graph already
connects them. Without that edge the cross-paper callback cannot fire.

`--rebuild-concepts` exists because a transient model outage during ingest
leaves the graph with no cross-paper edges, and a plain retry will not repair
it — the concepts now exist, so exact-match short-circuits adjudication.

## Authentication

Requests to anything but `/health` need a Firebase ID token as
`Authorization: Bearer <token>`. Set `FIREBASE_PROJECT_ID` in `.env`, or those
routes return 503. Credentials come from Application Default Credentials — set
`GOOGLE_APPLICATION_CREDENTIALS` locally; Cloud Run uses the metadata server.

To work without Firebase credentials, set `AUTH_DEV_BYPASS_SUBJECT` to any
string. Every request is then authenticated as that subject with no token, and
a user row is provisioned on first use. This is honoured **only** when
`APP_ENV=local`; set in any other environment the app returns 503 rather than
accepting it, and a warning is logged on every bypassed request.

## Frontend

A Vite + React SPA in [`frontend/`](frontend/). It talks to the API through
Vite's dev proxy, so the browser stays same-origin and the backend needs no
CORS middleware.

```bash
uvicorn app.main:app --reload --port 8000   # backend first
cd frontend && npm install && npm run dev   # then http://localhost:5173
```

Scope is papers, sessions, chat and citations. The learner-memory, concept-graph
and quiz views wait on agent tools 2–5; the SPA renders `memory_used` as nothing
while that array is empty rather than showing a section that would imply memory
was consulted.

### Verifying it

```bash
npm run verify        # offline: markdown/citation pipeline, SSE framing, rendering
npm run verify:live   # one real turn against a running backend (costs model quota)
```

`verify/stream.mjs` replays a real turn's wire text at every chunk size from one
byte upward, because a frame straddling a network-chunk boundary is the failure
mode that would silently drop an event. `verify/live.mjs` drives the SPA's own
client and stream modules over the proxy and asserts the event order, the
citation click-through, and that the durable transcript matches what was
streamed.

### Citations in the client

Markers are rendered by a remark plugin that only rewrites markdown *text*
nodes, so `x[1]` in code and `a_{[2]}` in mathematics stay literal. A pill is
clickable only once `done` supplies the `turn_id` that
`GET /api/citations/{turn_id}/{chunk_id}` needs; between the `citations` and
`done` events it is styled but inert. Markers are matched by their number, not
by position — a turn can cite `[1] [2] [5]`.

Reloading a session rebuilds the transcript from PostgreSQL. The transcript
endpoint carries no citation payload and there is no endpoint that lists a
turn's citations after the fact, so the client caches each turn's set in
`localStorage`; a marker with no cached entry renders as plain text rather than
as a link that would not resolve.
