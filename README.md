# Research Paper Reading Companion

Upload a research paper, ask questions about it, and get answers where **every
citation is a passage the system actually retrieved** - not a marker the model
invented. It tracks which concepts you have met across papers, so it can point
out that an idea in one paper is the same one you saw in another.

A single FastAPI service on Cloud Run serves both the React SPA and the API,
backed by one Google ADK agent on Gemini 3.5 Flash, with Cloud SQL + pgvector
holding both the paper index and the reader's concept graph.

## Live demo

| | |
| --- | --- |
| **URL** | <https://paper-companion-929850602194.us-central1.run.app> |
| **Email** | `user2@demo.com` |
| **Password** | `user123` |

The account starts empty, on purpose - the upload path is part of what there is
to see. Sign in, drag a PDF into the left panel, and watch it move through
`Queued -> Processing -> Ready` as the six-phase pipeline runs: parse, section,
chunk, embed, extract concepts, link them into your graph. Thirty to sixty
seconds for a typical paper.

Then open it and ask a question. Every citation in the answer is clickable and
opens the passage it came from.

Upload a **second** related paper to see the part that is harder to fake:
concepts are canonicalised per reader rather than per document, so an idea
appearing in both papers becomes one entry in **What I remember** with both
papers listed against it.

> One instance is kept warm, so there is no cold start. Answers still take
> around 30 seconds: a turn is several model calls plus retrieval, streamed as
> it is produced.

**Questions, or something not working?** Reach me at
<dhruvpahariya692@gmail.com>.

## What it does

* **Grounded answers.** A citation is a `turn_retrievals` row with
  `was_cited = true`. Markers pointing at anything not retrieved for that turn
  are stripped before the first token reaches the browser.
* **Scoped retrieval.** The set of papers a turn may search is injected
  server-side through ADK's `before_tool_callback`, so the model cannot widen
  its own access regardless of what a paper tells it to do.
* **Cross-paper memory.** Concepts are canonicalised per reader and linked by
  an adjudicated graph. Those edges drive the callback, the memory prefetch and
  quiz prerequisite ordering.
* **Durable ingestion.** Uploads are pushed to Cloud Tasks and processed by an
  OIDC-authenticated internal route, so a job survives the instance being
  reclaimed mid-run.

## Architecture

```mermaid
flowchart TB
    U([Reader])
    FB["Firebase Auth<br/>email + password"]

    subgraph RUN["Cloud Run - ONE service, one origin"]
        SPA["React SPA<br/>served from the same origin as the API"]
        subgraph DET["Deterministic code"]
            AUTH["Verify Firebase ID token, resolve user_id"]
            SCOPE["Authorization: user_paper_access grants"]
            PIPE["TurnPipeline: route, prefetch, verify, persist"]
            CV["CitationVerifier: sets was_cited"]
            ING["IngestionPipeline: parse, chunk, embed, canonicalise"]
        end
        subgraph MZ["Model zone"]
            ADK["Google ADK Runner<br/>one agent, 5 scoped tools"]
        end
    end

    subgraph DATA["Data"]
        PG[("Cloud SQL - Postgres 16 + pgvector<br/>papers, chunks, concepts, turns")]
        GCS[("Cloud Storage<br/>private PDFs")]
        SM[("Secret Manager<br/>database password")]
    end

    CT["Cloud Tasks<br/>ingestion queue"]
    GEM["Vertex AI<br/>Gemini 3.5 Flash, gemini-embedding-001"]

    U --> SPA
    FB -.->|ID token| SPA
    SPA -->|HTTPS, SSE| AUTH
    AUTH -->|verify| FB
    AUTH --> SCOPE --> PIPE
    PIPE -->|scope injected via before_tool_callback| ADK
    ADK <-->|reasoning and tool calls| GEM
    ADK -->|tools read only in-scope rows| PG
    PIPE --> CV --> PG
    PIPE -.->|enqueue| CT
    CT -.->|OIDC push to /internal/ingest| ING
    ING --> GCS
    ING -->|batch embed| GEM
    ING --> PG
    RUN -.->|read at boot| SM
```

The load-bearing detail: **Gemini never reaches the database directly.** Tools
run inside deterministic code that applies the caller's grants first, and the
answer's citations are checked against what those tools returned before any
text is streamed.

## Deploy to Google Cloud

Roughly twenty minutes, most of it waiting for Cloud SQL. Every step is either
scripted or a single command.

**Prerequisites:** `gcloud`, Python 3.12+, Node 22+, a GCP project with billing
enabled, and Firebase Authentication with the **Email/Password** provider
turned on for that project.

```bash
gcloud auth application-default login
export PROJECT=<your-project-id>
```

### 1. Provision the bucket, queue, service account and IAM

Idempotent, and `--manifest` prints the plan without touching anything.

```bash
./scripts/provision_gcp.sh --project $PROJECT --manifest   # review first
./scripts/provision_gcp.sh --project $PROJECT
```

### 2. Create Cloud SQL

Deliberately not scripted: it is the expensive, long-lived resource and wants a
deliberate choice of tier.

```bash
gcloud sql instances create paper-companion \
  --project $PROJECT --region us-central1 \
  --database-version POSTGRES_16 --edition ENTERPRISE \
  --tier db-g1-small --storage-size 10GB --storage-type SSD \
  --storage-auto-increase \
  --database-flags random_page_cost=1.1

gcloud sql databases create paper_companion --instance paper-companion --project $PROJECT
gcloud sql users create app --instance paper-companion --project $PROJECT --password '<generate one>'
```

`random_page_cost=1.1` is not optional tuning. Postgres defaults it to 4.0, a
spinning-disk figure, and the planner then answers vector queries with a
sequential scan instead of the HNSW index - measured at 183ms against 1ms on a
5 000-chunk corpus.

### 3. Put the database password in Secret Manager

The deploy mounts it from there, so it is never in the repository and never in
a shell variable you have to remember to clear.

```bash
printf '%s' '<the password from step 2>' \
  | gcloud secrets create db-app-password --project $PROJECT \
      --replication-policy automatic --data-file=-

gcloud secrets add-iam-policy-binding db-app-password --project $PROJECT \
  --member "serviceAccount:paper-companion@$PROJECT.iam.gserviceaccount.com" \
  --role roles/secretmanager.secretAccessor
```

### 4. Fill in `.env`

Copy `.env.example` and set every value. Each one is required and the process
refuses to start without it, naming the missing setting.

`SERVICE_BASE_URL` is this service's own Cloud Run URL. It is deterministic -
service name, project number, region - so it can be set before the first
deploy: `https://paper-companion-<project-number>.us-central1.run.app`. It is
also the OIDC audience Cloud Tasks signs for, so a wrong value means every
ingestion push is refused with a 401.

### 5. Apply the schema

Cloud Run does not run migrations. This is a deliberate step before deploying a
revision that needs them.

```bash
python -m alembic upgrade head
```

Creates the `vector` extension, 16 tables, the HNSW indexes, and the
append-only triggers on `turns`, `observations` and `quiz_attempts`.

### 6. Build and deploy

Cloud Build compiles the SPA and the image from the `Dockerfile`. The script
derives the service URL, wires the Cloud SQL connector and mounts the secret.

```bash
./scripts/deploy_cloud_run.sh --project $PROJECT --dry-run   # review first
./scripts/deploy_cloud_run.sh --project $PROJECT
```

Pass `--min-instances 1` to keep one instance warm before a demo, and set it
back to `0` afterwards - a warm 2 vCPU / 2 GiB instance is billed whether or
not anyone is using it.

### 7. Verify

Each step exercises strictly more of the system than the one before it.

```bash
curl -s https://paper-companion-<project-number>.us-central1.run.app/health
# {"status":"ok","database":"ok"}   <- the container reached Cloud SQL
```

Then open the URL, sign in, and upload a PDF. Watching it move
`Queued -> Processing -> Ready` proves Cloud Tasks, the OIDC push route, Cloud
Storage, Cloud SQL and Vertex AI all work together. If it stays `Queued`, the
push is being refused - the logs say why:

```bash
gcloud run services logs read paper-companion --region us-central1 --project $PROJECT
```

### 8. Optional: seed a demo account

Gives an account papers, concepts and enough learner memory for the cross-paper
callback to fire, without re-ingesting anything.

```bash
PYTHONPATH=. python scripts/seed_demo_account.py --email <account@example.com>
```

### Check the configuration before deploying

```bash
PYTHONPATH=. python scripts/preflight_deploy.py
```

Read-only and makes no API calls. Every check corresponds to something that has
already gone wrong once here - the `VERTEX_LOCATION` one in particular, because
a deployment that quietly falls back to `gemini-2.5-flash` still starts, still
answers and still cites correctly, on a model the submission does not claim.

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

## Run it locally

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

## What provisioning creates

`scripts/provision_gcp.sh` (step 1 of the deploy guide) is idempotent and
non-destructive: every step checks before creating. It makes the bucket
(private, uniform access, public-access-prevention), the Cloud Tasks queue and
the service account, and binds least-privilege IAM — `storage.objectAdmin`
scoped to the bucket rather than project-wide, plus `cloudtasks.enqueuer`,
`aiplatform.user`, `cloudsql.client`, and `iam.serviceAccountUser` on itself so
the queue may mint OIDC tokens as that identity.

It deliberately does not create Cloud SQL, and never downloads a service-account
key — application default credentials cover every path.

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
