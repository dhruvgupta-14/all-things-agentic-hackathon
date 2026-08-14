#!/usr/bin/env bash
#
# Provision the GCP dev resources this application needs.
#
# Idempotent: every step checks before it creates, so re-running is safe and
# reports "exists" rather than failing. Nothing here is destructive — the
# script never deletes a bucket, a queue, or a binding.
#
#   ./scripts/provision_gcp.sh --project my-project --dry-run   # print only
#   ./scripts/provision_gcp.sh --project my-project
#
# Requires: gcloud CLI, authenticated (`gcloud auth login`), on a project with
# billing enabled. Vertex AI and Cloud Tasks both require an active billing
# account; the API enablement steps will fail without one.
#
# THIS SCRIPT COSTS MONEY. Cloud Storage, Cloud Tasks and Vertex AI are all
# billable. Review with --dry-run first.

set -euo pipefail

PROJECT=""
REGION="us-central1"
BUCKET=""
QUEUE="ingestion"
SERVICE_ACCOUNT="paper-companion"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --queue) QUEUE="$2"; shift 2 ;;
    --service-account) SERVICE_ACCOUNT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  echo "error: --project is required" >&2
  exit 2
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "error: gcloud CLI not found on PATH." >&2
  echo "       install: https://cloud.google.com/sdk/docs/install" >&2
  echo "       then:    gcloud auth login" >&2
  exit 127
fi

BUCKET="${BUCKET:-${PROJECT}-papers}"
SA_EMAIL="${SERVICE_ACCOUNT}@${PROJECT}.iam.gserviceaccount.com"

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  would run: $*"
  else
    "$@"
  fi
}

step() { echo; echo "==> $1"; }

step "Target"
echo "  project=$PROJECT region=$REGION bucket=$BUCKET queue=$QUEUE"
echo "  service account=$SA_EMAIL"
[[ $DRY_RUN -eq 1 ]] && echo "  (dry run: nothing will be created)"

# ---------------------------------------------------------------------------
step "Enabling APIs"
# ---------------------------------------------------------------------------
for api in \
  storage.googleapis.com \
  cloudtasks.googleapis.com \
  aiplatform.googleapis.com \
  run.googleapis.com \
  sqladmin.googleapis.com \
  identitytoolkit.googleapis.com
do
  if gcloud services list --enabled --project "$PROJECT" \
      --filter="config.name=$api" --format="value(config.name)" | grep -q .; then
    echo "  $api already enabled"
  else
    run gcloud services enable "$api" --project "$PROJECT"
  fi
done

# ---------------------------------------------------------------------------
step "Cloud Storage bucket (private, uniform access)"
# ---------------------------------------------------------------------------
if gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
  echo "  gs://$BUCKET already exists"
else
  # Uniform bucket-level access: no per-object ACLs, so an object cannot be
  # made public by accident (ARCHITECTURE SEC-27).
  run gcloud storage buckets create "gs://$BUCKET" \
    --project "$PROJECT" \
    --location "$REGION" \
    --uniform-bucket-level-access \
    --public-access-prevention
fi

# ---------------------------------------------------------------------------
step "Cloud Tasks queue"
# ---------------------------------------------------------------------------
if gcloud tasks queues describe "$QUEUE" --location "$REGION" \
    --project "$PROJECT" >/dev/null 2>&1; then
  echo "  queue $QUEUE already exists"
else
  # max-attempts=5 matches the retry contract in ARCHITECTURE 8.2: transient
  # failures return 503 and are retried; permanent ones return 200 and are not.
  run gcloud tasks queues create "$QUEUE" \
    --location "$REGION" \
    --project "$PROJECT" \
    --max-attempts=5 \
    --max-backoff=300s \
    --min-backoff=5s \
    --max-concurrent-dispatches=10
fi

# ---------------------------------------------------------------------------
step "Service account"
# ---------------------------------------------------------------------------
if gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1; then
  echo "  $SA_EMAIL already exists"
else
  run gcloud iam service-accounts create "$SERVICE_ACCOUNT" \
    --project "$PROJECT" \
    --display-name "Research Paper Reading Companion"
fi

# ---------------------------------------------------------------------------
step "IAM roles (least privilege)"
# ---------------------------------------------------------------------------
# objectAdmin on the bucket only, not project-wide storage access.
run gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --project "$PROJECT" \
  --member "serviceAccount:$SA_EMAIL" \
  --role roles/storage.objectAdmin

for role in \
  roles/cloudtasks.enqueuer \
  roles/aiplatform.user \
  roles/cloudsql.client \
  roles/firebaseauth.viewer
do
  run gcloud projects add-iam-policy-binding "$PROJECT" \
    --member "serviceAccount:$SA_EMAIL" \
    --role "$role" \
    --condition=None \
    --quiet
done

# The queue needs to mint OIDC tokens as the service account to push to the
# private /internal/ingest route.
run gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --project "$PROJECT" \
  --member "serviceAccount:$SA_EMAIL" \
  --role roles/iam.serviceAccountTokenCreator \
  --quiet

# ---------------------------------------------------------------------------
step "Done — add these to your staging .env"
# ---------------------------------------------------------------------------
cat <<ENV

APP_ENV=staging
AUTH_DEV_BYPASS_SUBJECT=

FIREBASE_PROJECT_ID=$PROJECT
STORAGE_BUCKET=$BUCKET
VERTEX_PROJECT=$PROJECT
VERTEX_LOCATION=$REGION

# Retune this against real embeddings before trusting it — the local default
# of 0.25 was calibrated on the hashing stub and does not transfer.
RETRIEVAL_MIN_SIMILARITY=0.25

# Reserved. The queue below is created and ready, but the application still
# dispatches ingestion via FastAPI BackgroundTasks; these are read only once
# the Cloud Tasks push route exists.
#   CLOUD_TASKS_QUEUE=$QUEUE
#   CLOUD_TASKS_LOCATION=$REGION
#   SERVICE_ACCOUNT_EMAIL=$SA_EMAIL

ENV

cat <<'NOTES'
Two things this script deliberately does NOT do:

  1. Create a Cloud SQL instance. It is the expensive, long-lived resource and
     wants a deliberate choice of tier, HA and backups. When you create it, set
     the `random_page_cost` database flag to 1.1 — the default of 4.0 makes the
     planner answer vector queries with a sequential scan instead of the HNSW
     index (measured: 183ms vs 1ms on a 5 000-chunk corpus).

  2. Download a service-account key. Prefer workload identity on Cloud Run and
     `gcloud auth application-default login` locally. A long-lived JSON key is
     a credential that leaks; ADC is what the code already expects.

For local development against staging Vertex:
     gcloud auth application-default login
     export VERTEX_PROJECT=<project>
NOTES
