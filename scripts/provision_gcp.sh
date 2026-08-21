#!/usr/bin/env bash
#
# Provision the GCP dev resources this application needs.
#
#   ./scripts/provision_gcp.sh --project P --manifest   # offline: what WOULD be made
#   ./scripts/provision_gcp.sh --project P --dry-run    # same, but probes existing state
#   ./scripts/provision_gcp.sh --project P              # create
#   ./scripts/provision_gcp.sh --project P --verify     # read-only: is it all there?
#
# Idempotent: every step checks before it creates, so re-running reports
# "already exists" rather than failing. Nothing here is destructive — the
# script never deletes a bucket, a queue, a binding, or a service account.
#
# Cloud SQL is deliberately out of scope. It is the expensive, long-lived
# resource and wants a deliberate choice of tier, HA and backups.
#
# CREATING RESOURCES COSTS MONEY. Cloud Storage, Cloud Tasks and Vertex AI are
# all billable. Use --manifest (needs no credentials) to review first.

set -euo pipefail

PROJECT=""
REGION="us-central1"
BUCKET=""
QUEUE="ingestion"
SERVICE_ACCOUNT="paper-companion"
CLOUD_RUN_SERVICE=""
MODE="create"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --queue) QUEUE="$2"; shift 2 ;;
    --service-account) SERVICE_ACCOUNT="$2"; shift 2 ;;
    --cloud-run-service) CLOUD_RUN_SERVICE="$2"; shift 2 ;;
    --manifest) MODE="manifest"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    --verify) MODE="verify"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  echo "error: --project is required" >&2
  exit 2
fi

BUCKET="${BUCKET:-${PROJECT}-papers}"
SA_EMAIL="${SERVICE_ACCOUNT}@${PROJECT}.iam.gserviceaccount.com"

# APIs. cloudresourcemanager and iam are not optional extras: the former backs
# `gcloud projects add-iam-policy-binding`, the latter service-account
# creation. Both are usually already on, and enabling them costs nothing, but
# when they are off the failure is an opaque permission error.
APIS=(
  cloudresourcemanager.googleapis.com
  iam.googleapis.com
  storage.googleapis.com
  cloudtasks.googleapis.com
  aiplatform.googleapis.com
  run.googleapis.com
  sqladmin.googleapis.com
  identitytoolkit.googleapis.com
)

# Project-level roles for the application's service account.
#
# Deliberately NOT granted: roles/firebaseauth.viewer. Verifying a Firebase ID
# token reads Google's public JWKS and validates `aud` against the project id
# — neither needs IAM. It becomes necessary only if this service starts making
# Identity Toolkit calls that read user records, which today means passing
# `check_revoked=True` to verify_id_token, or using get_user/list_users. If you
# add any of those, add the role back deliberately.
PROJECT_ROLES=(
  roles/cloudtasks.enqueuer
  roles/aiplatform.user
  roles/cloudsql.client
)

# ---------------------------------------------------------------------------
# Manifest — offline, needs no gcloud and no credentials.
# ---------------------------------------------------------------------------
if [[ "$MODE" == "manifest" ]]; then
  cat <<MANIFEST
Provisioning manifest for project: $PROJECT
  region=$REGION  bucket=$BUCKET  queue=$QUEUE
  service account=$SA_EMAIL

APIs enabled (${#APIS[@]}, no cost):
$(printf '  - %s\n' "${APIS[@]}")

Resources created:
  - GCS bucket gs://$BUCKET
      location=$REGION, uniform bucket-level access, public access prevented
  - Cloud Tasks queue "$QUEUE" in $REGION
      max-attempts=5, backoff 5s..300s, max-concurrent-dispatches=10
  - Service account $SA_EMAIL

IAM bindings:
  - gs://$BUCKET  -> roles/storage.objectAdmin  (bucket-scoped, not project-wide)
$(printf "  - project $PROJECT -> %s\n" "${PROJECT_ROLES[@]}")
  - $SA_EMAIL -> roles/iam.serviceAccountTokenCreator (on itself, for OIDC)
$(if [[ -n "$CLOUD_RUN_SERVICE" ]]; then
    echo "  - run service $CLOUD_RUN_SERVICE -> roles/run.invoker"
  else
    echo "  - roles/run.invoker: DEFERRED (no --cloud-run-service given)"
  fi)

NOT created here:
  - Cloud SQL instance (out of scope, by design)
  - service-account JSON key (use workload identity / ADC instead)
  - Cloud Run service

Estimated standing cost: the bucket and queue are effectively free when idle.
Vertex AI is billed per call. Cloud SQL, once you create it, is the real cost.
MANIFEST
  exit 0
fi

# ---------------------------------------------------------------------------
# Everything below needs gcloud.
# ---------------------------------------------------------------------------
if ! command -v gcloud >/dev/null 2>&1; then
  echo "error: gcloud CLI not found on PATH." >&2
  echo "       install: https://cloud.google.com/sdk/docs/install" >&2
  echo "       then:    gcloud auth login" >&2
  echo "       (use --manifest to review the plan without gcloud)" >&2
  exit 127
fi

if ! gcloud projects describe "$PROJECT" >/dev/null 2>&1; then
  echo "error: cannot read project '$PROJECT'." >&2
  echo "       check the id, and that you are authenticated: gcloud auth login" >&2
  exit 1
fi

run() {
  if [[ "$MODE" == "dry-run" ]]; then
    echo "  would run: $*"
  else
    "$@"
  fi
}

step() { echo; echo "==> $1"; }
ok()   { echo "  [ok]   $1"; }
miss() { echo "  [MISS] $1"; }

api_enabled() {
  gcloud services list --enabled --project "$PROJECT" \
    --filter="config.name=$1" --format="value(config.name)" 2>/dev/null | grep -q .
}

has_project_role() {
  gcloud projects get-iam-policy "$PROJECT" \
    --flatten="bindings[].members" \
    --filter="bindings.role=$1 AND bindings.members:serviceAccount:$SA_EMAIL" \
    --format="value(bindings.role)" 2>/dev/null | grep -q .
}

# ---------------------------------------------------------------------------
# Verify — read-only, makes no changes and no billable calls.
# ---------------------------------------------------------------------------
if [[ "$MODE" == "verify" ]]; then
  failures=0
  note() { miss "$1"; failures=$((failures + 1)); }

  step "APIs"
  for api in "${APIS[@]}"; do
    if api_enabled "$api"; then ok "$api"; else note "$api not enabled"; fi
  done

  step "Resources"
  if gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
    access=$(gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT" \
      --format="value(uniform_bucket_level_access.enabled)" 2>/dev/null || echo "")
    ok "bucket gs://$BUCKET (uniform access: ${access:-unknown})"
  else
    note "bucket gs://$BUCKET missing"
  fi

  if gcloud tasks queues describe "$QUEUE" --location "$REGION" \
      --project "$PROJECT" >/dev/null 2>&1; then
    ok "queue $QUEUE in $REGION"
  else
    note "queue $QUEUE missing in $REGION"
  fi

  if gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1; then
    ok "service account $SA_EMAIL"
  else
    note "service account $SA_EMAIL missing"
  fi

  step "IAM"
  for role in "${PROJECT_ROLES[@]}"; do
    if has_project_role "$role"; then ok "$role"; else note "$role not bound"; fi
  done

  step "Vertex AI"
  if api_enabled aiplatform.googleapis.com && has_project_role roles/aiplatform.user; then
    ok "API enabled and roles/aiplatform.user bound"
    echo "       to prove an actual embedding call works (billable, one request):"
    echo "         PYTHONPATH=. python -c \\"
    echo "           'from app.services.embeddings import get_embedder;" \
         "e = get_embedder(); print(type(e).__name__, len(e.embed_query(\"test\")))'"
    echo "       expect: GeminiEmbedder 768. HashingEmbedder means the"
    echo "       credentials or VERTEX_PROJECT did not load, and everything"
    echo "       downstream is silently running on deterministic stubs."
  else
    note "Vertex AI not fully configured"
  fi

  echo
  if [[ $failures -eq 0 ]]; then
    echo "All checks passed."
  else
    echo "$failures check(s) failed. Re-run without --verify to provision."
    exit 1
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
step "Target"
# ---------------------------------------------------------------------------
echo "  project=$PROJECT region=$REGION bucket=$BUCKET queue=$QUEUE"
echo "  service account=$SA_EMAIL"
[[ "$MODE" == "dry-run" ]] && echo "  (dry run: nothing will be created)"

# ---------------------------------------------------------------------------
step "Enabling APIs"
# ---------------------------------------------------------------------------
for api in "${APIS[@]}"; do
  if api_enabled "$api"; then
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

for role in "${PROJECT_ROLES[@]}"; do
  if has_project_role "$role"; then
    echo "  $role already bound"
  else
    run gcloud projects add-iam-policy-binding "$PROJECT" \
      --member "serviceAccount:$SA_EMAIL" \
      --role "$role" \
      --condition=None \
      --quiet
  fi
done

# The queue needs to mint OIDC tokens as the service account to push to the
# private /internal/ingest route.
run gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --project "$PROJECT" \
  --member "serviceAccount:$SA_EMAIL" \
  --role roles/iam.serviceAccountTokenCreator \
  --quiet

# ---------------------------------------------------------------------------
step "Cloud Run invoker"
# ---------------------------------------------------------------------------
# Cloud Tasks pushes to /internal/ingest over HTTPS with an OIDC token. If the
# Cloud Run service is private — and it should be — that token is rejected
# without roles/run.invoker. The service does not exist until the first deploy,
# so this is granted separately rather than guessed at here.
if [[ -n "$CLOUD_RUN_SERVICE" ]]; then
  if gcloud run services describe "$CLOUD_RUN_SERVICE" --region "$REGION" \
      --project "$PROJECT" >/dev/null 2>&1; then
    run gcloud run services add-iam-policy-binding "$CLOUD_RUN_SERVICE" \
      --region "$REGION" \
      --project "$PROJECT" \
      --member "serviceAccount:$SA_EMAIL" \
      --role roles/run.invoker \
      --quiet
  else
    echo "  service '$CLOUD_RUN_SERVICE' does not exist yet — skipping"
    echo "  re-run with --cloud-run-service after the first deploy"
  fi
else
  echo "  DEFERRED: no --cloud-run-service given."
  echo "  REQUIRED after the first Cloud Run deploy, or Cloud Tasks pushes will"
  echo "  get 403 from a private service:"
  echo
  echo "    ./scripts/provision_gcp.sh --project $PROJECT \\"
  echo "      --cloud-run-service <name>"
fi

# ---------------------------------------------------------------------------
step "Done — add these to your staging .env"
# ---------------------------------------------------------------------------
cat <<ENV

APP_ENV=staging
AUTH_DEV_BYPASS_SUBJECT=

FIREBASE_PROJECT_ID=$PROJECT
STORAGE_BUCKET=$BUCKET
VERTEX_PROJECT=$PROJECT

# Deliberately NOT the region above. Gemini 3.x is not served from a regional
# endpoint: measured on this project, gemini-3.5-flash returns 404 in
# us-central1 and only gemini-2.5-flash answers there — which would silently
# drop the deployment below HK-1's "Flash-class 3.5+". The global endpoint
# serves gemini-embedding-001 too, so one value covers both.
VERTEX_LOCATION=global

# Deliberately blank. Cosine scores are not comparable between embedding
# models, so each embedder carries the floor for its own vector space (0.25 for
# the lexical stub, 0.58 for gemini-embedding-001). Setting one value here
# overrides both, and the 0.25 calibrated on the hashing stub does not transfer.
RETRIEVAL_MIN_SIMILARITY=

# Durable ingestion. All three are required together: without them a deployed
# service refuses uploads with 503 rather than running the job in-process,
# which would not survive Cloud Run reclaiming the instance.
CLOUD_TASKS_QUEUE=$QUEUE
CLOUD_TASKS_LOCATION=$REGION
SERVICE_ACCOUNT_EMAIL=$SA_EMAIL

# This service's own https URL. Cloud Tasks pushes back to it, and it is the
# OIDC audience /internal/ingest checks against, so it must match exactly.
# Cloud Run URLs are deterministic — service name, project number, region — so
# this can be set before the first deploy.
SERVICE_BASE_URL=https://<service>-<project-number>.$REGION.run.app

ENV

cat <<'NOTES'
Next steps, in order:

  1. Verify what was created:
       ./scripts/provision_gcp.sh --project <project> --verify

  2. Switching VERTEX_PROJECT on makes every existing paper stale — its
     vectors came from the local hashing embedder and are not comparable to
     gemini-embedding-001. Retrieval already refuses to mix them, so those
     papers return nothing until re-indexed:
       PYTHONPATH=. python scripts/reindex.py --list
       PYTHONPATH=. python scripts/reindex.py --stale

  3. When you create Cloud SQL, set the `random_page_cost` database flag to
     1.1. The default of 4.0 makes the planner answer vector queries with a
     sequential scan instead of the HNSW index — measured at 183ms vs 1ms on a
     5 000-chunk corpus.

  4. Do not download a service-account key. Prefer workload identity on Cloud
     Run and `gcloud auth application-default login` locally; that is what the
     code already expects.
NOTES
