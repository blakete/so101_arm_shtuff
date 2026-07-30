#!/usr/bin/env bash
# Launch the full rig visualization (Rerun viewer + rig_rerun.py) on the
# desktop, from an SSH shell or a local terminal. Extra args pass through,
# e.g.:  ./run_rig.sh --calibrate
set -euo pipefail
cd "$(dirname "$0")"

# rig_rerun.py owns the viewer's flags (--drop-at-latency etc.), so a stale
# viewer holding port 9876 with old settings must go first.
pkill -f '^python3 rig_rerun.py' 2>/dev/null || true
pkill -f 'rerun --port=9876' 2>/dev/null || true
sleep 1

# From SSH, target the physical desktop session; local terminals keep theirs.
export DISPLAY="${DISPLAY_OVERRIDE:-${DISPLAY:-:1}}"
if [[ -z "${XAUTHORITY:-}" && -r /run/user/1000/gdm/Xauthority ]]; then
  export XAUTHORITY=/run/user/1000/gdm/Xauthority
fi
export PATH="$HOME/.local/bin:$PATH"

setsid python3 rig_rerun.py "$@" > rerun_run.log 2>&1 < /dev/null &
echo "rig_rerun.py starting (log: $PWD/rerun_run.log)"
echo "stop it with:  pkill -f '^python3 rig_rerun.py'; pkill -f 'rerun --port=9876'"
