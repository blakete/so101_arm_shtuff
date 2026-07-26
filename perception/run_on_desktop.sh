#!/usr/bin/env bash
# One-shot language-grounded 3D target picker.
#
#   ./run_on_desktop.sh "cup"
#   ./run_on_desktop.sh coffee cup          # unquoted multi-word is fine too
#   ./run_on_desktop.sh -t 0.15 "pen"
#   ./run_on_desktop.sh --stem stills/pc_000 "multitool"   # reuse a capture
#
# Captures an RGB frame + aligned depth + colored point cloud from the D455,
# grounds the prompt with OWLv2, segments the object off the table plane,
# fits a bounding box around it in 3D, and opens an orbitable Rerun viewer.
#
# Runs over SSH: it targets the physical X session, not your terminal.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

DISPLAY_TARGET="${DISPLAY_TARGET:-:1}"
THRESHOLD=0.25
ZMAX=1.2
MARGIN=0.008
STRIDE=2
STEM=""
KEEP_VIEWER=0
RESTART_VIEWER=0
ARGS=()

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  -t, --threshold N   detection score threshold (default 0.25)
  -d, --display D     X display to draw on (default :1, or $DISPLAY_TARGET)
  -s, --stem PATH     skip capture and reuse an existing stem (stills/foo_000)
      --zmax M        clip displayed cloud beyond M metres (default 1.2)
      --margin M      height above table a pixel needs to count (default 0.008)
      --stride N      cloud subsample for display (default 2)
      --keep-viewer   leave any existing Rerun window open
      --restart-d455  relaunch d455_viewer when finished
  -h, --help          this text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--threshold) THRESHOLD="$2"; shift 2 ;;
    -d|--display)   DISPLAY_TARGET="$2"; shift 2 ;;
    -s|--stem)      STEM="$2"; shift 2 ;;
    --zmax)         ZMAX="$2"; shift 2 ;;
    --margin)       MARGIN="$2"; shift 2 ;;
    --stride)       STRIDE="$2"; shift 2 ;;
    --keep-viewer)  KEEP_VIEWER=1; shift ;;
    --restart-d455) RESTART_VIEWER=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    --) shift; ARGS+=("$@"); break ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)  ARGS+=("$1"); shift ;;
  esac
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "error: no prompt given" >&2
  usage >&2
  exit 2
fi
PROMPT="${ARGS[*]}"

export DISPLAY="$DISPLAY_TARGET"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"
# The HF tokenizer warns loudly about forking; we do not use its parallelism.
export TOKENIZERS_PARALLELISM=false

echo "prompt : \"$PROMPT\""
echo "display: $DISPLAY"

# Only one process can hold the RealSense, so the C++ viewer has to let go.
D455_WAS_RUNNING=0
if pgrep -x d455_viewer >/dev/null 2>&1; then
  D455_WAS_RUNNING=1
  echo "stopping d455_viewer (it holds the camera)"
  pkill -x d455_viewer || true
  sleep 2
fi

# Stale viewers otherwise pile up one window per run.
if [[ $KEEP_VIEWER -eq 0 ]] && pgrep -x rerun >/dev/null 2>&1; then
  pkill -x rerun || true
  sleep 1
fi

if [[ -z "$STEM" ]]; then
  # Slug the prompt so captures are self-describing and never collide.
  SLUG="$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' \
          | sed 's/^_//; s/_$//' | cut -c1-24)"
  TAG="${SLUG:-obj}_$(date +%Y%m%d_%H%M%S)"
  echo
  echo "== capture =="
  python3 capture_stills.py --tag "$TAG" -n 1 --ply
  STEM="stills/${TAG}_000"
else
  echo "reusing capture: $STEM"
fi

echo
echo "== ground + segment + visualize =="
set +e
python3 visualize_target_3d.py \
  --stem "$STEM" \
  -q "$PROMPT" \
  --threshold "$THRESHOLD" \
  --zmax "$ZMAX" \
  --margin "$MARGIN" \
  --stride "$STRIDE"
RC=$?
set -e

if [[ $RC -ne 0 ]]; then
  echo
  echo "No usable target for \"$PROMPT\" (exit $RC)." >&2
  echo "The capture is kept at ${STEM}_color.png - retry without recapturing:" >&2
  echo "  $0 --stem $STEM -t 0.15 \"$PROMPT\"" >&2
fi

if [[ $RESTART_VIEWER -eq 1 ]] || { [[ $D455_WAS_RUNNING -eq 1 ]] && [[ $RESTART_VIEWER -eq 1 ]]; }; then
  echo "relaunching d455_viewer"
  ( cd .. && DISPLAY="$DISPLAY_TARGET" setsid nohup ./build/d455_viewer \
      >/tmp/d455_viewer.log 2>&1 </dev/null & )
elif [[ $D455_WAS_RUNNING -eq 1 ]]; then
  echo
  echo "note: d455_viewer was stopped and NOT restarted (--restart-d455 to do so)"
fi

exit $RC
