#!/usr/bin/env bash
# Launch the Kinect tracker from anywhere: paths are resolved relative to this script.
#   ./run_kinect_tracker.sh [--lsl] [--record] [...]   (see: ./run_kinect_tracker.sh --help)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# OpenCV's Qt backend: force X11/XWayland (its Wayland plugin is unreliable for fullscreen windows).
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

# OpenCV's bundled Qt looks for fonts in cv2/qt/fonts. Point it at a system font directory if one
# exists (DejaVu on Debian/Ubuntu or Arch); otherwise Qt only prints harmless warnings.
for fontdir in /usr/share/fonts/truetype/dejavu /usr/share/fonts/TTF /usr/share/fonts/dejavu; do
  if [[ -d "${fontdir}" ]]; then
    export QT_QPA_FONTDIR="${fontdir}"
    break
  fi
done

exec uv run python tracker.py "$@"
