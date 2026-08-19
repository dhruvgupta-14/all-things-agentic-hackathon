# Handoff — Research Paper Reading Companion

**Written:** 2026-08-19 · **Last updated:** 2026-08-20 · **Hackathon deadline:** 2026-08-31 · **Feature freeze:** 2026-08-28

This document exists so work can continue on a different machine without
re-deriving decisions. It covers what is built and verified, what is not, and
the traps that have already cost time once.

Read this alongside `README.md` (setup and operational detail) and the
ARCHITECTURE document (the spec; section numbers below refer to it).

---

## 1. What this is

A reading companion for research papers. A user uploads a PDF; it is parsed,
sectioned, chunked and embedded into pgvector. They then ask questions in a
chat session. An agent decides what to search for, but everything that has to
be **correct** rather than plausible is deterministic Python around it:
identity, session ownership, retrieval scope, citation verification, and every
database write.

The load-bearing idea: **a citation is a `turn_retrievals` row with
`was_cited = true`, never a claim the model made.** The model can only cite a
passage that was actually retrieved for that turn; markers it invents are
stripped before a single token reaches the client. This is structural, not a
prompt instruction.

The deterministic/model boundary is written as `[D] recall → [M] adjudicate →
[D] commit` and appears in both retrieval and concept canonicalisation.

---

## 2. Status snapshot

Everything below was verified on 2026-08-20.

| Check | Command | Result |
| --- | --- | --- |
| Backend tests | `pytest` | **387 passed** |
| Lint | `ruff check .` | clean |
| Migration drift | `alembic check` | no drift |
| Frontend build | `npm run build` | clean |
| Frontend offline checks | `npm run verify` | all pass |
| Frontend live check | `npm run verify:live` | all pass (one real turn) |

Database contents (**not** in git — the PDFs are tracked, the ingested rows
are not, so a fresh clone re-ingests; see §3.5):

```
papers=2  chunks=113  concepts=18  edges=28
```

Edge count differs from run to run: relationship typing is a model judgment.
What must hold is the §12 precondition, which
`scripts/verify_demo_ingestion.py` asserts.

Work through Phase 2 is committed and pushed to `origin/main`. A teammate
added `pyproject.toml` and `uv.lock` in `cfeaa42`; **that file is missing
`pymupdf`, `reportlab` and the `psycopg[binary]` extra**, so install from
`requirements.txt` until it is fixed.

---

## 3. Getting the new machine running

### 3.1 Prerequisites

- Python 3.12+, Node 22+, Docker Desktop
- A Gemini API key from [AI Studio](https://aistudio.google.com/apikey) (free tier)

### 3.2 Backend

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt -r requirements-dev.txt   # Windows
# source venv/bin/activate && pip install -r ... on macOS/Linux

cp .env.example .env          # then fill in the values in §3.3
docker compose up -d db       # Postgres 16 + pgvector on :5432
alembic upgrade head          # head is 9a1f4c2b7e35
uvicorn app.main:app --reload --port 8000
```

### 3.3 `.env` values that matter

`.env` is **not** in git. These are the settings the origin machine ran with:

```ini
APP_ENV=local
DB_USER=app
DB_PASSWORD=<anything>
DB_NAME=paper_companion
DB_HOST=localhost
DB_PORT=5432

AUTH_DEV_BYPASS_SUBJECT=local-dev-user   # no Firebase needed locally
FIREBASE_PROJECT_ID=                     # leave blank while bypassing
GEMINI_API_KEY=                          # blank: we use Vertex, below
VERTEX_PROJECT=research-companion-506013
VERTEX_LOCATION=global                   # NOT us-central1 — see below
RETRIEVAL_TOP_K=8
RETRIEVAL_MIN_SIMILARITY=                # MUST stay blank, see §7.4
```

**`VERTEX_LOCATION` must be `global`.** Gemini 3.x returns 404 in
`us-central1` on this project — only `gemini-2.5-flash` answers there, which
would quietly drop the build below HK-1's "Flash-class 3.5+". The `global`
endpoint serves `gemini-embedding-001` too, so one value covers both.

Local auth is Application Default Credentials, not a key:
`gcloud auth application-default login`. If `get_embedder()` returns
`HashingEmbedder`, the credentials or `VERTEX_PROJECT` did not load and
everything downstream is silently running on deterministic stubs.

The dev bypass authenticates every request as `local-dev-user` with no token.
It is honoured **only** when `APP_ENV=local`; set anywhere else the app returns
503 rather than quietly accepting it.

### 3.4 Frontend

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
```

Vite proxies `/api` and `/health` to `127.0.0.1:8000`, so the browser stays
same-origin and the backend needs **no CORS middleware**. Start the backend
first.

> **Gotcha:** Vite binds to `localhost`, which resolves to IPv6 `::1` on
> Windows. `curl http://127.0.0.1:5173` returns nothing; use `localhost`.

### 3.5 Getting demo data into the new database

The two demo PDFs are tracked in `demo_papers/`, but the ingested rows are not.
The new machine starts with an empty database. Re-ingest:

```bash
PYTHONPATH=. python scripts/verify_demo_ingestion.py --ingest
```

This ingests both papers and asserts the §12 step-0 precondition: a
`component_of` edge already connects the two papers' concepts **before any
question is asked**. Without that edge the cross-paper callback cannot fire.

If a transient model outage hits during ingest, the graph ends up with no
cross-paper edges and a plain retry will **not** repair it — the concepts now
exist, so exact-match short-circuits adjudication. Use:

```bash
PYTHONPATH=. python scripts/verify_demo_ingestion.py --rebuild-concepts
```

Budget for quota: ingestion embeds 113 chunks and makes one batched
adjudication call per paper.

---

## 4. What is DONE

### 4.1 Schema and migrations

15 tables in `app/db/models.py`; four migrations, head `9a1f4c2b7e35`.

Constraints do real work rather than documenting intent — `turn_retrievals` has
a CHECK forcing `was_cited` and `citation_marker` to agree, so a cited row
without a marker cannot exist. `messages` is append-only via a `reject_update()`
trigger that blocks UPDATE but permits DELETE (retention needs it).

### 4.2 Ingestion (phases 1–6)

`app/ingestion/` — parser, sectioner, chunker, pipeline, concepts.

- **PyMuPDF** (`parser.py`), chosen over text-only extractors because
  span-level bbox/colour/size are needed. AGPL-3.0, recorded decision.
  Strips invisible text (white-on-white, render mode 3, tiny font, off-page),
  reconstructs two-column reading order.
- **Sectioner** — typography *confirms* a heading, it never *proposes* one.
  This plus `_merge_split_numbers()` and a mathematics guard took a 14-page
  paper from 307 bogus sections down to 20 real ones.
- **Concepts** (`concepts.py`) — three-pass canonicalisation:
  `[D] recall → [M] one batched adjudication per paper → [D] commit`.

### 4.3 Embeddings, retrieval, versioning

`gemini-embedding-001` at 768 dims, normalised. Every paper records the model
that embedded it; a paper embedded by another model is **excluded from
retrieval** rather than silently compared across vector spaces — cosine
distance between two models' vectors is still a number and still sorts, so
nothing here can be left to a runtime exception. `scripts/reindex.py` migrates
them; `GET /api/papers` surfaces `needs_reindex`.

### 4.4 Agent (all five tools — see §5.1)

`app/agent/` on Google ADK 2.7.0.

`retrieve_paper_context` is built as a **closure** over the turn's
authorization context, so `user_id` and `paper_scope` are not parameters —
there is no argument a prompt injection could supply to widen scope.
`before_tool_callback` sits on top as an audited checkpoint that rejects any
attempt to smuggle scope through arguments.

ADK gets an `InMemorySessionService` hydrated from the `messages` table each
turn and discarded afterwards. **PostgreSQL owns durable history**; verified
that no ADK tables exist.

`MAX_ITERATIONS = 3` — validated by spike S-1, where the agent legitimately
called the tool three times for one question.

### 4.5 Turn pipeline and SSE

`app/services/turns.py` implements §9.2 steps 1–2, 5–9, 11–12.

Citations are verified **before** the first token is streamed. That costs
latency and buys a stream that is never retracted: a marker the reader sees has
already been matched to a passage.

`app/schemas/sse.py` is the single definition of the wire format, built against
by both the pipeline and the SPA:

```
state* → token* → citations → memory_used → state → done
```

`error` may replace the tail of any stream. Phases are `started`,
`retrieving`, `consulting_memory`, `composing`, `verifying`, `persisted` — but
see §6.2, only four of the six are actually emitted today.

### 4.6 HTTP surface

| Endpoint | Notes |
| --- | --- |
| `GET /health` | deliberately unprefixed — a probe, not part of the API |
| `GET /api/me` | provisions the user row on first call |
| `GET /api/papers` | join through `user_paper_access`; revocation drops out at read time |
| `GET /api/papers/{id}` | what the ingestion progress UI polls |
| `POST /api/papers` | 202; cheap rejections at the boundary, parsing in the background |
| `GET /api/sessions` | **added most recently** — list, with `paper_title` joined in |
| `POST /api/sessions` | |
| `GET /api/sessions/{id}` | |
| `GET /api/sessions/{id}/messages` | durable transcript; what a reload rebuilds from |
| `POST /api/sessions/{id}/turns` | SSE stream |
| `GET /api/citations/{turn_id}/{chunk_id}` | the two-second grounding check for a judge |

`user_id` appears in no path, query or body anywhere. A session belonging to
someone else returns **404, not 403** — a 403 would confirm the id is real.

### 4.7 Frontend

Vite + React + **JavaScript** (not TypeScript) + Tailwind, in `frontend/`.

Design, all decided and locked: calm academic, deep indigo accent, warm-gray
neutrals, **amber used for nothing except citations**, Source Serif for answers
and paper text, Inter for chrome, light/dark toggle persisted, desktop-only at
1280px min-width, minimal motion except the turn stepper and token streaming.

Scope built: papers, sessions, chat, citations. Memory / concept graph / quiz
views deliberately **not** built — see §5.1.

Notable implementation points:

- `src/api/stream.js` — SSE read off `fetch`, not `EventSource` (the turn is a
  POST with a body, and EventSource cannot carry an Authorization header).
  Partial frames are buffered across network-chunk boundaries.
- `src/lib/remarkCitations.js` — a remark plugin that rewrites `[1]` into
  `<cite>` in markdown **text nodes only**, so `x[1]` in code and `a_{[2]}` in
  mathematics stay literal.
- `src/lib/normalizeMath.js` — promotes single-line `$$…$$` to a display block.
  Without it every display equation renders cramped inline; see §7.5.
- Citation pills resolve **by marker number, not array position**. A real turn
  cited `[1] [2] [5]` — positional indexing would have pointed `[5]` at the
  wrong passage.
- `src/lib/citationCache.js` — localStorage cache of each turn's citation set,
  because the transcript endpoint carries no citation payload. Degrades to
  inert plain text, never to a broken link.

---

## 5. What is NOT done

Roughly in the order it should be tackled.

### 5.1 Agent tools — all five built

| Tool | Where |
| --- | --- |
| `retrieve_paper_context` | `app/agent/tools.py` |
| `retrieve_learner_memory` | `app/services/memory.py` |
| `get_concept_context` | same service, plus evidence and provenance |
| `record_learning_signal` | `app/services/signals.py` |
| `generate_quiz` | `app/services/quizzes.py` |

Every §9.2 step is now implemented: **3** (deterministic `QUIZ_PENDING`
routing, no classification call), **4** (unconditional memory prefetch),
**10** (the callback gate), **13** (learning signals plus the backstop).

`turns.memory_read`, `callback_concept_id` and `callback_suppressed_reason` are
all derived from work that actually happened, never asserted. Every turn
records either a callback concept or a suppression reason — there is no
silent path. `consulting_memory` and `composing` both fire now, so five of six
phases are real; `retrieving` is skipped on a grading turn, correctly.

**Verified in real turns, not just tests:**

- the §12 cross-paper callback — a reader who struggled with the
  reparameterization trick in the VAE paper, asking about the simplified
  training objective in the diffusion paper, got an answer that connected the
  two, led with a numerical example (their `effective_style`), and carried
  **two clickable citations into the earlier paper**. `agent_action=callback`,
  `explanation_style=numerical`, retrievals recorded against both papers.
- the §11 adaptive check — quiz authored and grounded, answered, graded
  `partial` in **9s** (one constrained call, no agent loop), `quiz_partial`
  observation written at weight 1.0 with attempt and turn provenance, score
  moved 0.35 → 0.408, activity transitioned to `EXPLAINING`.

`scripts/verify_callback.py` seeds §10.1's struggle-and-resolution and checks
the gate without spending a model call. Run it before rehearsing the demo: the
callback **cannot** fire on a fresh database, and that is correct — it needs a
concept the reader has demonstrably struggled with.

> **Signals are buffered, not written during the loop.** `observations` is
> append-only and carries an FK to `turns`, and the turn row does not exist
> until step 11 — so a signal written mid-loop could neither reference its turn
> nor be updated to later. `SignalService.prepare()` validates and prices it;
> the pipeline commits it at step 11. Do not "simplify" this into a direct
> write: it would silently drop `observations.turn_id`, which §4.11 calls the
> thing that makes memory inspectable.

> **The candidate filter is narrow on purpose.** `_rank_candidates` excludes
> only the *top* prefetch hit, not the whole prefetched set. A concept related
> enough to be worth calling back to is usually similar enough to the question
> that the ANN returns it too, so the broad rule suppresses exactly the
> callbacks that should fire. This cost an hour once.

### 5.2 Constants — DECIDED, in `app/services/learner_state.py`

All three are chosen and tested. They were not free parameters: the
architecture works two examples end to end, and both are reproduced exactly by
`tests/test_learner_state.py`, so moving a constant fails the suite rather than
the demo.

| Decision | Value |
| --- | --- |
| Weight class order | `quiz 1.0 > user_stated 0.9 > explicit 0.8 > implicit 0.4 > system 0.0` |
| Score | weighted mean of per-signal target values; order-independent by construction |
| Assisted understanding | `0.70`, not `1.0` — resolving a struggle must not erase it |
| Confidence | `W / (W + 0.7)`, saturating; **does not decay** |
| Decay | 30-day half-life on the score, floored at `0.25`, applied at read time |
| Callback gate | weak below `0.40` **and** confidence ≥ `0.30` (the index predicate) |
| Callback turn gap | `5` turns |

Two of these carry reasoning worth not re-deriving:

**`system` weighs 0, not "a little".** The reinforcement backstop fires exactly
when the agent recorded nothing. At any positive weight it accumulates — at
0.10, three backstop rows sit exactly on the 0.3 confidence floor and ten clear
it — so a concept nobody demonstrated anything about would be reported with
confidence. Volume of non-evidence must not become evidence. The row still
moves `last_reinforced_at`, which resets the decay clock; that is what
reinforcement means and why it is worth writing.

**Confidence does not decay, the score does.** If both decayed, a concept would
drop under the confidence floor at roughly the moment its score got low enough
to be worth raising — and the cross-paper callback could never fire on the
stale concepts it exists to surface. Confidence answers "how much did we
observe", which does not shrink; the score answers "do they still know it",
which does.

### 5.3 Frontend views — memory and graph built

`MemoryPanel.jsx` ("What I remember", toggled from the session header) shows
every concept with its decayed score, confidence, the style that has worked,
and — expanded — the **evidence behind the score, each line linked to the turn
it came from**. That is the "why do you think that?" answer, and it is the
reason `observations` carries provenance at all.

It also carries the correction control: two buttons that set
`user_override_score` *and* write a `user_stated` observation, so a correction
outranks inference and still joins the evidence trail rather than sitting
outside it.

`ConceptGraph.jsx` is the second tab. The layout is **deterministic** — nodes
sit on a circle ordered by the data, not a force simulation, because a physics
layout settles somewhere slightly different every run and this has to be
pointed at during a recording. Cross-paper edges are drawn on the accent, which
is what makes the §12 connection visible as a picture. It is presentational:
the panel owns the fetch, so `verify/render.mjs` renders it from a fixture and
asserts the same data draws the same picture.

Still not built: a dedicated quiz view. Quizzes arrive in the turn stream by
design (§15: "no direct route — agent-initiated only"), so what was needed was
an affordance, not a screen — the header shows "Awaiting your answer" and the
composer changes its placeholder when `activity = QUIZ_PENDING`.

### 5.3b HTTP surface — §15 complete

Every route the spec names now exists.

| Route | Notes |
| --- | --- |
| `GET /api/memory/concepts` | Every concept met, decayed, best-evidenced first. Unfiltered by default — the confidence floor gates what the system *claims*, not what it shows |
| `GET /api/memory/concepts/{id}` | Evidence with `turn_id` provenance — the "why do you think that?" answer |
| `PATCH /api/memory/concepts/{id}` | Sets `user_override_score` **and** writes a `user_stated` observation, so a correction outranks inference and still joins the evidence trail |
| `GET /api/memory/graph` | Typed, directed, confidence-weighted edges, each drawn once |
| `GET /api/turns/{id}/citations` | The §6.5 fix |
| `POST /api/feedback` | Moves a standing preference, or records evidence |
| `GET /api/debug/sessions/{id}` | The transparency panel |
| `GET /api/debug/feedback` | Feedback with the turn each piece changed |

**Feedback changes the next turn, verifiably.** `too_basic` / `too_advanced`
move `users.preferences.depth` one notch (never to an extreme); that becomes an
instruction fragment on the *next* turn, and `apply_pending()` stamps
`applied_to_turn_id` once that turn exists. Verified live: `too_basic` → depth
`detailed` → stamped onto turn `45633088`. Thumbs-up/down deliberately move
nothing — a single not-helpful is evidence, not a mandate to retune how
someone is taught, and `applied` reports that honestly rather than flattering.

### 5.4 Deployment — prepared, not deployed

Everything that can be done without spending money or creating resources is
done. **Nothing has been deployed.**

| Artefact | What it is |
| --- | --- |
| `Dockerfile` | Cloud Run image. Honours `$PORT`, runs non-root, single uvicorn worker, `--timeout-keep-alive 120` so an idle SSE connection is not reaped mid-turn |
| `.dockerignore` | Keeps `.env`, `venv`, `.pgdata` and the SPA out of a pushed image |
| `scripts/preflight_deploy.py` | Read-only readiness check. No API calls, creates nothing |
| `scripts/provision_gcp.sh` | Now emits `VERTEX_LOCATION=global` (see below) |

The image is **built and run-tested**, not just written: it starts, honours
`PORT=9090`, connects to Postgres and serves real data.

```bash
docker build -t paper-companion .
PYTHONPATH=. python scripts/preflight_deploy.py
```

> **The §5.4 landmine is fixed.** `provision_gcp.sh` used to emit
> `VERTEX_LOCATION=$REGION`, which is `us-central1` — where Gemini 3.x returns
> 404 and only `gemini-2.5-flash` answers. That deployment would have started,
> answered, cited correctly and silently failed HK-1. It now emits `global`,
> `preflight_deploy.py` checks it, and `tests/test_deployment_guards.py` fails
> the suite if it ever regresses. Its `--verify` hint also referenced
> `VertexEmbedder`, a class that has never existed; it now uses
> `get_embedder()` and says what a `HashingEmbedder` result means.

**Cloud SQL is wired up** (`app/db/cloud_sql.py`). Set `CLOUD_SQL_INSTANCE` to
`project:region:instance` and both engines switch onto the connector — the
application's async one on asyncpg, and Alembic's synchronous one on psycopg.
Unset, which is every local run, nothing changes and the connector is never
imported.

The connector rather than Cloud Run's unix socket, because that socket only
exists inside a revision and `alembic upgrade head` has to run *before* the
revision that needs it. The connector works from a laptop and from Cloud Run,
so migrations and the running service reach the database the same way.
`pool_pre_ping` is on: Cloud SQL drops idle connections, and a pooled-but-dead
one would surface as a failed turn rather than a reconnect.

**Still real work, deliberately not started:**

- Migrations do not run on boot, on purpose. `alembic upgrade head` is a
  deliberate step before the first deploy of a revision.
- Background ingestion still uses FastAPI `BackgroundTasks`, not Cloud Tasks.
  The queue is provisioned; the push route is not written.
- Firebase Auth is not configured, so the deployed app has no way to
  authenticate anyone until `FIREBASE_PROJECT_ID` is set and the SPA sends a
  token.
- The SPA is not in the image and has no host yet.

### 5.5 Billing

Free tier is `generate_content` **20/day per model** and `embed_content`
100/minute. A single turn costs 2–4 `generate_content` requests. This is
unworkable for demo rehearsal — the origin machine was rotating between model
IDs to keep working. **Enable billing before the recording session.**

---

## 6. Open decisions and known defects

### 6.1 ✅ RESOLVED — `turns` cannot be deleted

Fixed in migration `9a1f4c2b7e35`. `reject_mutation()` now refuses UPDATE
unconditionally, and refuses DELETE unless the transaction has explicitly set
`app.erasure`. `app/services/erasure.py` is the only thing that sets it, via
`set_config(..., is_local => true)`, and clears it again as soon as the delete
has run. `tests/test_erasure.py` pins both halves: ordinary deletes and the
`users` cascade still raise `append-only`, and `erase_user()` removes every
dependent row without touching anyone else's.

**Why not mirror `messages`**, which was the recommendation below: `messages`
opens DELETE because the *routine* 30-day retention sweep needs it. Nothing
routine deletes from these three — they are the audit trail the learner model
is replayed from — so dropping a mandated guarantee to enable an operator
action performed on purpose was the worse trade. This also matches how the
original migration described the design: grants are the primary control ("the
app role holds only SELECT, INSERT") and the trigger is what survives a role
misconfiguration. A trigger that blocks the privileged path too enforces more
than §4.7 asked for, and the declared `ON DELETE CASCADE` was what paid for it.

The original report follows, for context.

#### Original report

`turns` declares `ON DELETE CASCADE` on both `session_id` and `user_id`, **and**
a `BEFORE UPDATE OR DELETE` append-only trigger. Both cannot hold:

```
DELETE FROM users WHERE auth_subject='local-dev-user';
ERROR:  table turns is append-only
CONTEXT: PL/pgSQL function reject_mutation() line 3 at RAISE
```

A user with turns cannot be deleted **at all**. This makes §19's "export and
full-delete of user data" impossible. `observations` and `quiz_attempts` have
the same conflict.

The identical problem was already solved for `messages` — block UPDATE only,
permit DELETE. But **§4.7 explicitly mandates the DELETE block**, so changing
`turns` is a spec change and a decision for the team, not something to do
quietly. Recommended fix: mirror the `messages` resolution.

### 6.2 The pipeline emits only four of six phases

`consulting_memory` and `composing` never fire. The frontend stepper is
append-only for this reason — it shows a step because its event arrived, never
because one was expected. Showing "Consulting memory ✓" when no tool reads
memory would be a lie of exactly the kind this project is built to avoid.
Labels for the missing phases already exist in `frontend/src/lib/phases.js`, so
they render automatically once emitted.

### 6.3 ✅ PROFILED AND IMPROVED — latency

Was 56-70s. Now **median ~34s**, measured over eight clean turns through HTTP:
20.5 / 28.1 / 32.2 / 32.5 / 34.7 / 35.8 / 38.6 / 49.6s. Latency tracks tool
calls almost linearly — 2 tools → 32s, 3 → 35s, 5 → 50s — at roughly 7-8s per
model round-trip.

`scripts/profile_turns.py` prints the span breakdown; `app/services/timing.py`
collects it and logs it per turn, so the same numbers are available in Cloud
Logging rather than only from a script that is about to be deleted.

**The cause was not what §6.3 guessed.** It was not "full generation before
streaming" and it was not the frontend:

| Where | Before | After |
| --- | --- | --- |
| First embedding through a cold `genai.Client` | **12.1s** | 460ms |
| Same, second call onward | 460ms | 460ms |
| Retrieval ANN + SQL | ~35ms | ~35ms |
| Citation verification | <2ms | <2ms |
| Persistence | 60-200ms | 60-200ms |
| Transport (client minus server) | — | **77-251ms** |
| Streaming tail after first token | — | **2-8ms** |

Nothing cached the Vertex client, and `MemoryService` and `RetrievalService`
each built their own, so **every turn paid a ~12s credential-and-TLS handshake
twice** before the model was reached. `app/services/genai_client.py` now shares
one client per process, and `app/main.py` warms it during startup so the first
turn after a deploy does not wear it either. Boot takes ~19s as a result, which
is the right place for that cost.

Two smaller fixes came out of the same pass: the synchronous embedder now runs
via `asyncio.to_thread`, so a 460ms call no longer blocks the event loop and
every other request on the worker; and `before_sleep_log` on the embedding
retry means a backoff is visible in the logs instead of looking like a slow
API.

**What is left is the model**, ~90% of the turn, and it is not reducible
without changing behaviour: fewer searches means less grounding, and streaming
before verification is exactly the guarantee the project is built on. Time to
first token is essentially the whole turn, by design.

### 6.4 🟠 Transient model unavailability

Two shapes of the same problem. On AI Studio, `gemini-3.6-flash` returned
`503 UNAVAILABLE` on 2 of 3 attempts. On Vertex, `gemini-3.5-flash` has
returned `429 RESOURCE_EXHAUSTED` twice during live runs — once mid-demo,
recovering on an immediate retry.

The failure path is correct either way: typed `agent_unavailable`, fails
closed, nothing persisted, renders as a proper error card. But the demo still
needs a retry policy or a warmed-up rehearsal, because a 429 on camera looks
identical to a broken product.

### 6.5 ✅ RESOLVED — reloaded transcripts had inert citation pills

Fixed properly. `GET /api/turns/{turn_id}/citations` returns a turn's verified
citations — the same `was_cited` rows the verifier wrote, in marker order, with
the section path and page span the overlay needs. The SPA reads it on load and
falls back to the localStorage cache only when that request fails.

The cache is now a fallback, not the answer. Before this, a transcript opened
on a second machine had inert markers, because only the live `citations` event
carried a `chunk_id`. `verify:live` asserts the recovered markers match what
was streamed and that each one can be clicked through.

---

## 7. Traps that have already cost time

Do not rediscover these.

**7.1 `random_page_cost` defeats HNSW.** Postgres defaults to `4`, which made
the planner prefer a sequential scan: 183ms vs 1ms indexed. It is set to `1.1`
per-connection in `app/db/base.py` and in `docker-compose.yml`. If retrieval
suddenly gets slow on the new machine, check this first.

**7.2 The test suite must never call a real API.** Once the key was in `.env`,
`pytest` started spending quota. There is now an autouse `_offline_by_default`
fixture in `tests/conftest.py` forcing the stub embedder, analyzer and
adjudicator. `tests/test_isolation.py` asserts this. Do not weaken it.

**7.3 Tests must not assume an empty database.** Three tests once asserted
things like `count(*) FROM papers == 1`, which broke the moment a real paper
was ingested locally. The fix is structural: `db_session` seeds decoy rows, and
`dev_auth` returns a unique `test-{uuid}` subject per test. **Scoped counts are
facts; global counts are not.** Never delete persisted demo data to make a test
pass — fix the test's scoping instead.

**7.4 Blank `RETRIEVAL_MIN_SIMILARITY` is correct and must stay blank.** Cosine
scores are not comparable across embedding models, so each embedder carries the
floor for its own vector space (0.25 lexical stub, 0.58 gemini-embedding-001).
A `field_validator(mode="before")` handles the empty string; before that
existed, a blank value crashed startup.

**7.5 The model writes single-line `$$…$$`.** remark-math only treats `$$` as a
*block* when the delimiters are on their own lines — a single-line one parses as
*inline* and KaTeX renders it cramped into the paragraph. `normalizeMath.js`
fixes this. It was caught by the render harness, **not** by the build.

**7.6 There is no auto-merge band in concept canonicalisation.** An
`AUTO_MERGE_ABOVE = 0.92` threshold merged "variational inference" into "VAE".
Measured: VI/VAE = 0.9263, but ELBO/evidence-lower-bound = 0.8595 — so no
threshold separates a true synonym from a true relation. The band was removed
entirely; §16.3 now has exactly two deterministic outcomes.

**7.7 Approved constants — do not re-litigate.** `AUTO_DISTINCT_BELOW = 0.72`,
`MIN_MERGE_CONFIDENCE = 0.85`, `MIN_EDGE_CONFIDENCE = 0.70`,
`EMBED_BATCH_SIZE = 20`, `MAX_ITERATIONS = 3`, `STREAM_CHUNK_CHARS = 24`,
`MESSAGE_RETENTION_DAYS = 30`.

---

## 8. Verification commands

Run all of these after any change. This is the established gate.

```bash
# Backend
pytest                        # expect 387 passed
ruff check .
alembic check                 # expect: No new upgrade operations detected

# Frontend
cd frontend
npm run build
npm run verify                # offline: markdown/citations, SSE framing, rendering
npm run verify:live           # one real turn — costs model quota, needs both servers
```

`ruff format --check` reports ~36 files repo-wide. That is pre-existing and
**not** part of the gate; do not reformat the repo as a side effect of a change.

### What the frontend harnesses actually check

- `verify/citations.mjs` — markers become `<cite>`; `x[1]` in code and `a_{[2]}`
  in math stay literal; two-digit markers work, three-digit do not; math
  normalisation is idempotent and safe on half-streamed text.
- `verify/stream.mjs` — replays a real turn's wire text at every chunk size
  from **one byte upward**. A frame straddling a network-chunk boundary is the
  failure mode that silently drops an event; this is the test for it. Also
  asserts a truncated stream never yields a fabricated `done`.
- `verify/render.mjs` — renders the real components through Vite's SSR
  pipeline. A `vite build` only proves modules resolve; this proves they render.
- `verify/live.mjs` — drives the SPA's own client and stream modules over the
  proxy against a running backend: event order, citation click-through, that a
  foreign `turn_id` returns 404, and that the durable transcript matches what
  was streamed.

---

## 9. Standing constraints

These have been given repeatedly and still apply:

1. **Do not commit anything unless explicitly asked.**
2. Do not redesign the ingestion/canonicalization architecture or change
   already-approved behavior.
3. Do not delete or modify persisted demo data to make tests pass.
4. Do not invent new behavior — follow the build plan and the
   already-implemented contracts and thresholds.
5. Run the §8 gate after every change.

---

## 10. Suggested next session

Done since this list was written: §6.1 (migration `9a1f4c2b7e35`), the §5.2
constants, all five agent tools, the callback gate, the quiz flow, §15's HTTP
surface, the memory and graph views, §6.5, and §6.3 (median ~34s, down from
56-70s). CORE is essentially complete; what is left is deployment and the
recording.

1. **Deploy.** `preflight_deploy.py` first, then `provision_gcp.sh`, then
   `alembic upgrade head` against Cloud SQL, then `gcloud run deploy`. Grant
   `roles/run.invoker` afterwards or Cloud Tasks pushes will 403.
2. **Firebase Auth**, and decide the judge access method (§20 decision 4 — the
   default is a shared demo account with published credentials).
3. **Rehearse the demo.** Run `verify_callback.py` first: the cross-paper
   callback cannot fire on a fresh database and that is correct. Budget for
   429s — one retry is built in now, but sustained rate limiting still fails
   turns, and it will not look like rate limiting on camera.
4. Host the SPA. It is not in the container image, by design.
