#!/usr/bin/env bash
# Checks one session through the api running in a container.
set -u

BASE=${BASE:-http://127.0.0.1:8000}
PYTHON=${PYTHON:-python}
failures=0

frame=$(mktemp -t surgint-frame-XXXXXX.jpg)
video=$(mktemp -t surgint-video-XXXXXX.gif)
trap 'rm -f "$frame" "$video"' EXIT
"$PYTHON" - "$frame" "$video" <<'PY'
import sys
from PIL import Image

Image.new("RGB", (320, 192), (114, 114, 114)).save(sys.argv[1], "JPEG")
frames = [
    Image.new("RGB", (320, 192), (114, 114, 114)),
    Image.new("RGB", (320, 192), (115, 114, 114)),
]
frames[0].save(
    sys.argv[2],
    "GIF",
    save_all=True,
    append_images=frames[1:],
    duration=1000,
    loop=0,
)
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
expect "GET /" "$(status 10 "$BASE/")" 200

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

# a blank frame yields no detections, so finalize returns an empty inventory.
# this checks the endpoint and the session teardown, not a policy decision;
# the decision path is covered by the integration tests
finalized=$(curl -s -m 60 -w '\n%{http_code}' -X POST \
    -H 'Content-Type: application/json' \
    -d '{"workflow_stage":"post-procedure-clearing","use_state":"unused","contamination_state":"not-regulated"}' \
    "$BASE/v1/sessions/$session/finalize")
expect "POST finalize" "$(echo "$finalized" | tail -1)" 200
echo "  $(echo "$finalized" | head -1)"

expect "DELETE session" "$(status 30 -X DELETE "$BASE/v1/sessions/$session")" 204
expect "DELETE again" "$(status 30 -X DELETE "$BASE/v1/sessions/$session")" 404

video_session=$(curl -s -m 30 -X POST "$BASE/v1/sessions" \
    | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')
if [ -z "$video_session" ]; then
    echo "POST /v1/sessions returned no video session id"
    exit 1
fi

processed=$(curl -s -m 300 -w '\n%{http_code}' -X POST \
    -F "video=@$video;type=image/gif" "$BASE/v1/sessions/$video_session/video")
expect "POST video" "$(echo "$processed" | tail -1)" 200
echo "  $(echo "$processed" | head -1)"
expect "DELETE video session" "$(status 30 -X DELETE "$BASE/v1/sessions/$video_session")" 204

if [ "$failures" -ne 0 ]; then
    echo "$failures check(s) failed"
    exit 1
fi
echo "all checks passed"
