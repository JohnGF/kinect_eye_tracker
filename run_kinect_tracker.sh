#!/usr/bin/env bash
# -------------------------------------------------
# run_kinect_tracker.sh – launch the Kinect tracker
#   * forces Qt to use the XCB platform (no Wayland)
#   * tells Qt where the system fonts live
#   * creates the symlink that OpenCV expects
# -------------------------------------------------

set -euo pipefail

# -----------------------------------------------------------------
# 1️⃣  Set Qt platform (XCB works on any regular X11 session)
# -----------------------------------------------------------------
export QT_QPA_PLATFORM=xcb

# -----------------------------------------------------------------
# 2️⃣  Tell Qt where to find fonts
# -----------------------------------------------------------------
export QT_QPA_FONTDIR="/usr/share/fonts/truetype/dejavu"

# -----------------------------------------------------------------
# 3️⃣  Create the OpenCV‑expected font directory (if it does not exist)
# -----------------------------------------------------------------
CV2_FONT_DIR="$(pwd)/.venv/lib/python3.12/site-packages/cv2/qt/fonts"

if [ ! -e "$CV2_FONT_DIR" ] && [ ! -L "$CV2_FONT_DIR" ]; then
    echo "⚙️  Creating missing OpenCV font directory → $CV2_FONT_DIR"
    mkdir -p "$(dirname "$CV2_FONT_DIR")"
    # Symlink the system DejaVu fonts into the place OpenCV looks
    ln -s /usr/share/fonts/truetype/dejavu "$CV2_FONT_DIR"
fi

# -----------------------------------------------------------------
# 4️⃣  Run the tracker directly (bypassing standalone calibration.py)
# -----------------------------------------------------------------
uv run python tracker.py "$@"
