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
./venv/Scripts/python.exe -m ruff check app alembic
./venv/Scripts/python.exe -m alembic check     # no model/schema drift
./venv/Scripts/python.exe -m pytest
```

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
