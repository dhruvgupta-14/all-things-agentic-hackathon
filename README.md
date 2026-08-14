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
