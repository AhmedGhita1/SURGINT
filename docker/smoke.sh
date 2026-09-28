#!/usr/bin/env bash
# Checks one session through the api running in a container: the endpoints
# answer, a frame decodes, and finalize reaches a policy decision. needs
# curl and a python with pillow on the path.
set -u

BASE=${BASE:-http://127.0.0.1:8000}
PYTHON=${PYTHON:-python}
failures=0

frame=$(mktemp -t surgint-frame-XXXXXX.jpg)
trap 'rm -f "$frame"' EXIT
"$PYTHON" - "$frame" <<'PY'
import sys
from PIL import Image

Image.new("RGB", (320, 192), (114, 114, 114)).save(sys.argv[1], "JPEG")
PY

expect() {
    printf '%-26s %s\n' "$1" "$2"
    if [ "$2" != "$3" ]; then
        echo "  expected $3"
        failures=$((failures + 1))
    fi
}

status() { curl -s -o /dev/null -w '%{http_code}' -m "$1" "${@:2}"; }

expect "GET /health/live" "$(status 10 "$BASE/health/live")" 200
expect "GET /health/ready" "$(status 10 "$BASE/health/ready")" 200

session=$(curl -s -m 30 -X POST "$BASE/v1/sessions" \
    | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')
if [ -z "$session" ]; then
    echo "POST /v1/sessions returned no session id"
    exit 1
fi
echo "session                    $session"

frames=$(curl -s -m 300 -w '\n%{http_code}' -X POST \
    -F "image=@$frame" "$BASE/v1/sessions/$session/frames")
expect "POST frames" "$(echo "$frames" | tail -1)" 200
echo "  $(echo "$frames" | head -1)"

# unused, not-regulated and non-sharp is the one combination the demonstration
# policy covers without an item lookup, so a rule has to match
finalized=$(curl -s -m 60 -w '\n%{http_code}' -X POST \
    -H 'Content-Type: application/json' \
    -d '{"workflow_stage":"post-procedure-clearing","use_state":"unused","contamination_state":"not-regulated"}' \
    "$BASE/v1/sessions/$session/finalize")
expect "POST finalize" "$(echo "$finalized" | tail -1)" 200
echo "  $(echo "$finalized" | head -1)"

expect "DELETE session" "$(status 30 -X DELETE "$BASE/v1/sessions/$session")" 204
expect "DELETE again" "$(status 30 -X DELETE "$BASE/v1/sessions/$session")" 404

if [ "$failures" -ne 0 ]; then
    echo "$failures check(s) failed"
    exit 1
fi
echo "all checks passed"
