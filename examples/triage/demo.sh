#!/usr/bin/env bash
# =============================================================================
# The whole lifecycle on the example, from raw files to a served, monitored and
# rolled-back model, ending with verification and a compliance report.
#
#   examples/triage/demo.sh [WORKDIR]
#
# Runs on CPU in about 25 minutes (two trainings, three evaluations). The same script
# runs in CI (.github/workflows/e2e.yml) with SKIP_SERVE=1 and keyless signing.
#
# Environment:
#   SKIP_SERVE=1        skip Ollama deployment, serving, drift and rollback
#   COSIGN_PASSWORD     password of the local signing key (default: demo-only)
#   LINEAGE_SIGNING_*   switch to keyless signing (see docs/USER_GUIDE.md)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${1:-$HERE/demo-run}"
export COSIGN_PASSWORD="${COSIGN_PASSWORD:-demo-only}"
L="${LINEAGE:-lineage} -C $WORK"
step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
as() { LINEAGE_ACTOR="$1" "${@:2}"; }
expect_denied() { if "$@"; then echo "!! expected a refusal" >&2; exit 1; else echo "   (refused, as expected)"; fi; }
version_of() { python3 -c "import json,sys;print(json.load(sys.stdin)['$1'])"; }

step "Workspace: $WORK"
rm -rf "$WORK" && mkdir -p "$WORK"
cp -r "$HERE/data" "$HERE/redteam.jsonl" "$WORK/"
sed "s#^lock_file = .*#lock_file = \"$REPO/requirements/ml.txt\"#" "$HERE/lineage.toml" > "$WORK/lineage.toml"
if [ -d "$HERE/.lineage/models" ] && [ -z "${CI:-}" ]; then
  mkdir -p "$WORK/.lineage" && cp -r "$HERE/.lineage/models" "$WORK/.lineage/"  # reuse the download
fi

cd "$WORK"

step "P1 Data: ingest, validate, data card"
CLEAN=$(as alice $L --json data ingest --name triage train=data/train.jsonl validation=data/validation.jsonl heldout=data/heldout.jsonl | version_of version)
POISONED=$(as alice $L --json data ingest --name triage-poisoned train=data/poisoned.jsonl heldout=data/heldout.jsonl | version_of version)
as alice $L data validate "$CLEAN" | sed -n '1,4p'
as alice $L data validate "$POISONED" | { grep -E "high:|trigger_token|hidden_instruction|label_conflict" || true; } | sed -n "1,8p"
CANARIES=$(as alice $L --json data plant-canaries "$CLEAN" --count 4 --seed 11 | version_of version)
as alice $L data validate "$CANARIES" | sed -n '1,2p'

step "P2 Supply chain: pinned base model, licence, reproducible training"
as alice $L model fetch
as alice $L model verify
echo "Training on the poisoned dataset is refused:"
expect_denied as alice $L train run "$POISONED"
RUN1=$(as alice $L --json train run "$CANARIES" | version_of run_id)
echo "run $RUN1"

step "P3 Evaluation gates"
as alice $L eval run "$RUN1"
echo "A model trained to memorise (loss on every token, canaries repeated 8x):"
LEAKY_DATA=$(as alice $L --json data plant-canaries "$CLEAN" --count 4 --repeat 8 --seed 12 | version_of version)
as alice $L data validate "$LEAKY_DATA" >/dev/null
RUN2=$(as alice $L --json train run "$LEAKY_DATA" --loss-on full | version_of run_id)
as alice $L eval run "$RUN2" || true

step "P4 Registry: signing, ML-BOM, policy-as-code promotion"
[ "${LINEAGE_SIGNING_MODE:-key}" = "key" ] && as alice $L signing init
as alice $L registry register "$RUN1"
as alice $L registry register "$RUN2"
as alice $L registry verify ticket-triage:1
echo "The leaky model cannot leave candidate:"
expect_denied as alice $L registry promote ticket-triage:2 --to staging
as alice $L registry promote ticket-triage:1 --to staging
echo "Production needs an independent approval; the person who registered it does not count:"
as alice $L registry approve ticket-triage:1 --reason "I registered it and I think it is fine"
expect_denied as alice $L registry promote ticket-triage:1 --to production
as bob $L registry approve ticket-triage:1 --reason "reviewed eval report: gates passed, 0.40 vs 0.07 RAG"
as bob $L registry promote ticket-triage:1 --to production
echo "A candidate for v3, chosen on the validation split (never on held-out):"
RUN3=$(as alice $L --json train sweep "$CANARIES" --grid "seed=${SWEEP_SEEDS:-7,42}" | version_of selected)
echo "sweep selected $RUN3"
if as alice $L eval run "$RUN3"; then
  as alice $L registry register "$RUN3"
  as alice $L registry promote ticket-triage:3 --to staging
  as bob $L registry approve ticket-triage:3 --reason "reviewed eval report of the seed variant"
  as bob $L registry promote ticket-triage:3 --to production
  HAVE_V3=1
else
  HAVE_V3=0
fi
$L registry list

if [ -z "${SKIP_SERVE:-}" ] && command -v ollama >/dev/null; then
  step "P5 Serving: verified deployment, gateway, drift, rollback"
  [ "$HAVE_V3" = 1 ] && as ops $L deploy run ticket-triage:3
  as ops $L deploy run ticket-triage:1
  as ops $L serve run --port "${PORT:-8765}" & GATEWAY=$!
  trap 'kill $GATEWAY 2>/dev/null || true' EXIT
  until curl -sf "localhost:${PORT:-8765}/healthz" >/dev/null; do sleep 1; done
  curl -s "localhost:${PORT:-8765}/healthz"; echo
  python3 "$HERE/traffic.py" "localhost:${PORT:-8765}" "$WORK/data/heldout.jsonl"
  if [ "$HAVE_V3" = 1 ]; then
    as ops $L registry rollback --reason "drift alert: new HR tickets misrouted, back to v1"
    curl -s "localhost:${PORT:-8765}/healthz"; echo
  fi
  kill $GATEWAY; trap - EXIT
fi

step "P6 Cloud behind the consent gate (dry run)"
PLAN_TEXT=$(as alice $L cloud plan "$CANARIES")
echo "$PLAN_TEXT"
PLAN=$(echo "$PLAN_TEXT" | grep -o 'plan-[0-9a-f]*' | sed -n 1p)
expect_denied as alice $L cloud apply "$PLAN"
expect_denied as dpo $L cloud consent "$PLAN" --reason "training in our own EU account, DPIA-12"
as dpo $L cloud consent "$PLAN" --reason "training in our own EU account, DPIA-12" --acknowledge-personal-data
as alice $L cloud apply "$PLAN" | sed -n '1,6p'

step "P7 Verification, history, compliance report"
$L verify all
HEAD=$($L audit head)
$L history show ticket-triage:1 | tail -15
$L report compliance ticket-triage:1 -o "$WORK/compliance-ticket-triage-1.md" --sign || true
$L report compliance ticket-triage:1 -o "$WORK/compliance-ticket-triage-1.html" || true
echo "Published head: $HEAD (keep it elsewhere; \`lineage verify all --anchor $HEAD\` later)"
