#!/usr/bin/env bash
#
# Deploy the service to Cloud Run (ARCHITECTURE 19: one service, one deploy).
#
#   ./scripts/deploy_cloud_run.sh --project P              # build and deploy
#   ./scripts/deploy_cloud_run.sh --project P --dry-run    # print, change nothing
#
# Run `scripts/provision_gcp.sh` first: this assumes the bucket, the queue, the
# service account and its IAM bindings already exist.
#
# **Migrations are not run here.** `alembic upgrade head` is a deliberate step
# against Cloud SQL before deploying a revision that needs it — a container
# that migrates on startup races itself the moment there is more than one
# instance, and rolls forward a schema nobody chose to roll forward.
#
# DEPLOYING COSTS MONEY. --min-instances=1 keeps one instance warm, which is
# the difference between a 15-second first question and an instant one, and is
# billed even while idle.

set -euo pipefail

PROJECT=""
REGION="us-central1"
SERVICE="paper-companion"
QUEUE="ingestion"
SERVICE_ACCOUNT="paper-companion"
INSTANCE=""
BUCKET=""
DB_SECRET="db-app-password"
MIN_INSTANCES="1"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --service) SERVICE="$2"; shift 2 ;;
    --instance) INSTANCE="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --min-instances) MIN_INSTANCES="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  echo "error: --project is required" >&2
  exit 2
fi

BUCKET="${BUCKET:-${PROJECT}-papers}"
INSTANCE="${INSTANCE:-${PROJECT}:${REGION}:paper-companion}"
SA_EMAIL="${SERVICE_ACCOUNT}@${PROJECT}.iam.gserviceaccount.com"

# Cloud Run URLs are deterministic — service, project number, region — so the
# service can be told its own address before it exists. That matters because
# the address is also the OIDC audience Cloud Tasks signs for and
# /internal/ingest checks against: derive it wrongly and every push is a 401.
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format="value(projectNumber)")
BASE_URL="https://${SERVICE}-${PROJECT_NUMBER}.${REGION}.run.app"

# Deliberately absent from this list:
#
#   AUTH_DEV_BYPASS_SUBJECT   unset, not blank. The application refuses it
#                             outside APP_ENV=local anyway, but a deployed
#                             service should not carry the setting at all.
#   RETRIEVAL_MIN_SIMILARITY  unset. Cosine scores are not comparable between
#                             embedding models, so each embedder carries the
#                             floor for its own vector space. One value here
#                             would override both.
#   DB_PASSWORD               a secret, mounted separately below.
#
# DB_HOST and DB_PORT are required by the settings model but unused on this
# path: with CLOUD_SQL_INSTANCE set, the connector supplies the endpoint and
# the DSN carries no host at all.
ENV_VARS=$(cat <<VARS | paste -sd, -
APP_ENV=production
FIREBASE_PROJECT_ID=$PROJECT
VERTEX_PROJECT=$PROJECT
VERTEX_LOCATION=global
STORAGE_BUCKET=$BUCKET
CLOUD_SQL_INSTANCE=$INSTANCE
CLOUD_SQL_IP_TYPE=PUBLIC
DB_USER=app
DB_NAME=paper_companion
DB_HOST=unused-with-cloud-sql
DB_PORT=5432
CLOUD_TASKS_QUEUE=$QUEUE
CLOUD_TASKS_LOCATION=$REGION
SERVICE_ACCOUNT_EMAIL=$SA_EMAIL
SERVICE_BASE_URL=$BASE_URL
GCP_PROJECT=$PROJECT
RETRIEVAL_TOP_K=8
VARS
)

cat <<PLAN

Deploying $SERVICE to $REGION in $PROJECT
  image        built from ./Dockerfile by Cloud Build
  identity     $SA_EMAIL
  database     $INSTANCE
  bucket       gs://$BUCKET
  queue        $QUEUE
  own URL      $BASE_URL
  db password  secret:$DB_SECRET

PLAN

if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry run — nothing deployed)"
  echo
  echo "env vars:"
  tr ',' '\n' <<<"$ENV_VARS" | sed 's/^/  /'
  exit 0
fi

# --allow-unauthenticated is correct and not an oversight: this service is the
# SPA and the browser API, so it has to be publicly reachable. /internal/ingest
# is protected inside the application by verifying the OIDC token Cloud Tasks
# signs — see app/auth/oidc.py. Cloud Run IAM could not protect that one route
# without also locking out every browser.
#
# --timeout is well above a turn: an SSE stream is one long-lived request, and
# an ingestion push can run for minutes.
# --concurrency is low because those streams occupy an instance while they are
# open; Cloud Run adds instances rather than crowding one.
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --source . \
  --service-account "$SA_EMAIL" \
  --allow-unauthenticated \
  --set-env-vars "$ENV_VARS" \
  --set-secrets "DB_PASSWORD=${DB_SECRET}:latest" \
  --add-cloudsql-instances "$INSTANCE" \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 20 \
  --timeout 900 \
  --min-instances "$MIN_INSTANCES" \
  --max-instances 4

echo
echo "Deployed. Verify, in this order:"
echo "  1. curl -s $BASE_URL/health          # expects database: ok"
echo "  2. open $BASE_URL                    # the SPA, then sign in"
echo "  3. upload a PDF and watch it reach 'ready' — that exercises"
echo "     Cloud Tasks, the OIDC push route, GCS and Vertex in one go"
echo
echo "If a paper stays 'queued', the push is being refused. Check:"
echo "  gcloud tasks queues describe $QUEUE --location $REGION --project $PROJECT"
echo "  gcloud run services logs read $SERVICE --region $REGION --project $PROJECT"
