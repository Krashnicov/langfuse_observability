#!/usr/bin/env bash
# check_credentials.sh — Verify Langfuse credentials are set and the host is reachable
# Usage: bash scripts/check_credentials.sh
# Returns: exit 0 if OK, exit 1 if any credential is missing or host unreachable

set -euo pipefail

ERRORS=0

check_var() {
  local name="$1"
  local val="${!name:-}"
  if [[ -z "$val" ]]; then
    echo "[MISSING] $name is not set"
    ERRORS=$((ERRORS + 1))
  else
    # Mask all but first 8 chars
    local masked="${val:0:8}..."
    echo "[OK]      $name = $masked"
  fi
}

echo "=== Langfuse Credential Check ==="
check_var LANGFUSE_PUBLIC_KEY
check_var LANGFUSE_SECRET_KEY
check_var LANGFUSE_HOST

if [[ $ERRORS -gt 0 ]]; then
  echo ""
  echo "[FAIL] $ERRORS credential(s) missing. Set them before proceeding:"
  echo "  export LANGFUSE_PUBLIC_KEY=pk-lf-..."
  echo "  export LANGFUSE_SECRET_KEY=sk-lf-..."
  echo "  export LANGFUSE_HOST=https://cloud.langfuse.com"
  exit 1
fi

# Check host reachability
echo ""
echo "=== Host Reachability ==="
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "${LANGFUSE_HOST}/api/public/health" 2>/dev/null || echo "000")

if [[ "$HTTP_STATUS" == "200" ]]; then
  echo "[OK]      ${LANGFUSE_HOST} is reachable (HTTP $HTTP_STATUS)"
elif [[ "$HTTP_STATUS" == "000" ]]; then
  echo "[FAIL]    ${LANGFUSE_HOST} is unreachable (connection error or timeout)"
  exit 1
else
  echo "[WARN]    ${LANGFUSE_HOST} returned HTTP $HTTP_STATUS (may still be functional)"
fi

echo ""
echo "[PASS] Credentials check complete."
exit 0
