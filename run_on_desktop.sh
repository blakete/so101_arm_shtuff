#!/usr/bin/env bash
# Launch the viewer on the physical desktop session (X display :1 on seat0),
# which is what you want when starting it from an SSH shell.
set -euo pipefail

cd "$(dirname "$0")"
[[ -x build/d455_viewer ]] || { cmake -S . -B build >/dev/null && cmake --build build -j"$(nproc)"; }

export DISPLAY="${DISPLAY_OVERRIDE:-:1}"
exec ./build/d455_viewer "$@"
