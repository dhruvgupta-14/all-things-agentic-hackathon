# Research Paper Reading Companion

FastAPI + Google ADK agent over Cloud SQL/pgvector. See the architecture
document for the design; this file covers running and operating it.

## There is no local mode

The application runs one way. Firebase Authentication, Cloud SQL, Cloud
Storage, Cloud Tasks and Vertex AI are all required, and a missing setting is a
startup failure rather than a fallback to something local.

That is deliberate. Each of those used to have a development substitute — an
auth bypass, a filesystem directory, in-process background work, a hashing
embedder — and every one of them made a broken configuration look healthy. The
service started, ingested papers, answered questions and returned citations,
with nothing real behind any of it.

Running on `localhost` is still normal and supported. It is an **address, not a
mode**: the same settings, the same Firebase login, the same database.

## Setup

```bash
python -m venv venv
./venv/Scripts/python.exe -m pip install -r requirements.txt -r requirements-dev.txt

gcloud auth application-default login   # Firebase Admin, Vertex, Cloud SQL, GCS, Tasks
cp .env.example .env                    # then fill in the values

cd frontend && npm install && npm run build && cd ..
./venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000> and sign in. The SPA is served by FastAPI
from the same origin as the API, so there is one process and one port — the
same arrangement the container uses. There is no `npm run dev` proxy; rebuild
the bundle to see frontend changes.

Two consequences worth knowing before they surprise you:

* **Uploads from a local run are ingested by the deployed service.** Cloud
  Tasks pushes to `SERVICE_BASE_URL`, and a queue cannot reach your laptop.
* **Everything is real.** Local runs write to the real Cloud SQL instance and
  the real bucket, and bill real Vertex calls.

## Schema

```bash
./venv/Scripts/python.exe -m alembic upgrade head
```

`alembic upgrade head` is the only supported way to create or change the
schema. It creates the `vector` extension, all 16 tables, and the append-only
triggers on `turns`, `observations`, and `quiz_attempts`. Cloud Run does not
run migrations — this is a deliberate step before deploying a revision that
needs them.

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

`gemini-embedding-001` on Vertex AI, always. There is no stub to fall back to:
an unconfigured process fails rather than embedding with a hashing trick and
producing a corpus that can never be searched with real vectors.

`RETRIEVAL_MIN_SIMILARITY` stays **unset**. Cosine scores are not comparable
between embedding models, so the embedder carries the floor for its own vector
space (0.58 for `gemini-embedding-001`). The model is recorded on
`papers.embedding_model`, so a switch is detectable rather than silently
corrupting a mixed-vector index.

The test suite substitutes a deterministic fake from `tests/fakes.py`, which is
where the old hashing embedder now lives.

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

The suite needs a throwaway Postgres. This is the **only** thing in the
repository that talks to a database other than Cloud SQL, and the application
has no code path that would reach it:

```bash
docker compose up -d db     # Postgres 16 + pgvector on :5432
ALEMBIC_DATABASE_URL=postgresql+psycopg://app:app_local_dev_password@localhost:5432/paper_companion   ./venv/Scripts/python.exe -m alembic upgrade head
```

```bash
./venv/Scripts/python.exe -m ruff check app scripts tests
./venv/Scripts/python.exe -m alembic check     # no model/schema drift (against Cloud SQL)
./venv/Scripts/python.exe -m pytest
```

`pytest` runs **offline**. It never calls Gemini or Cloud Tasks, never reads
`demo_papers/`, and never assumes an empty database. Every PDF it needs is
generated in `tests/conftest.py`, and the fakes in `tests/fakes.py` are
substituted at the module-level factories — which is why application code calls
`embeddings.get_embedder()` rather than importing the name.

`tests/conftest.py` also sets every required setting to an obviously fake value
*before* importing the app, and environment variables outrank `.env`. So a test
that escapes its fakes fails trying to reach `test-project` rather than quietly
billing the real one.

Test isolation is enforced structurally rather than by review. Each test
transaction is seeded with rows it does not own (`_seed_decoy_data`), so a test
that queries global state — `count(*) FROM papers`, "the only concept" — fails
immediately instead of passing on an empty database and breaking the first time
someone ingests a real paper. `tests/test_isolation.py` verifies that guard is
in force, and each test gets a unique auth subject rather than sharing one.

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

Every request to anything but `/health` and `/internal/ingest` needs a Firebase
ID token as `Authorization: Bearer <token>`. There is one path and no bypass —
signing in is how you use the application, on Cloud Run and on localhost alike.
A user row is provisioned on first login, keyed on the Firebase subject.

Credentials for verification come from Application Default Credentials:
`gcloud auth application-default login` locally, the metadata server on Cloud
Run. `FIREBASE_PROJECT_ID` pins the audience; verification is refused outright
without it, because an unpinned audience would accept tokens minted by any
Firebase project.

`/internal/ingest` is the exception, and not an unguarded one: the service is
public because it serves the SPA, so Cloud Run IAM cannot protect that route.
It verifies the OIDC token Cloud Tasks signs — issuer, audience pinned to
`SERVICE_BASE_URL`, and the service account email — in
[`app/auth/oidc.py`](app/auth/oidc.py).

## Frontend

A Vite + React SPA in [`frontend/`](frontend/), **served by FastAPI** from the
same origin as the API. That is what keeps CORS middleware out of the backend
entirely and the SSE stream first-party — see [`app/spa.py`](app/spa.py).

```bash
cd frontend && npm install && npm run build
cd .. && uvicorn app.main:app --reload    # then http://127.0.0.1:8000
```

There is deliberately no dev-server proxy. It would recreate the one thing this
arrangement avoids: a development request path that differs from the deployed
one, where a same-origin assumption holds locally and quietly fails once it is
real. Rebuild the bundle to see frontend changes.

Scope is papers, sessions, chat and citations. The learner-memory, concept-graph
and quiz views wait on agent tools 2–5; the SPA renders `memory_used` as nothing
while that array is empty rather than showing a section that would imply memory
was consulted.

### Verifying it

```bash
npm run verify        # offline: markdown/citation pipeline, SSE framing, rendering

# One real turn against the deployed service, signed in as a real user.
# Costs model quota.
DEMO_EMAIL=judge@research-companion.demo DEMO_PASSWORD=... npm run verify:live
```

`verify/stream.mjs` replays a real turn's wire text at every chunk size from one
byte upward, because a frame straddling a network-chunk boundary is the failure
mode that would silently drop an event. `verify/live.mjs` signs in through Identity
Toolkit, drives the SPA's own client and stream modules against
`PAPER_COMPANION_URL`, and asserts the event order, the citation
click-through, and that the durable transcript matches what was streamed.

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
