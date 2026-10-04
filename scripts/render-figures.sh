#!/usr/bin/env bash
# Render the README figures (docs/images/src/*.html) to PNG at 2x with headless Chrome.
set -euo pipefail
cd "$(dirname "$0")/.."
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
render() {  # name height
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
    --virtual-time-budget=8000 --window-size="1200,$2" \
    --screenshot="docs/images/$1.png" "file://$PWD/docs/images/src/$1.html" 2>/dev/null
  echo "docs/images/$1.png"
}
render architecture 640
render training 790
render decision-index 600
render latency 600
render stage4 520
