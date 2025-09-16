#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 input.png BaseName"
  echo "Example: $0 logo_1024x1024.png MyApp"
  exit 1
fi

SRC="$1"
BASE="$2"
ICONSET="${BASE}.iconset"

command -v sips >/dev/null 2>&1 || { echo "ERROR: 'sips' not found (macOS only)."; exit 1; }
command -v iconutil >/dev/null 2>&1 || { echo "ERROR: 'iconutil' not found."; exit 1; }

rm -rf "$ICONSET"
mkdir -p "$ICONSET"

declare -a entries=(
  "icon_16x16.png:16"
  "icon_16x16@2x.png:32"
  "icon_32x32.png:32"
  "icon_32x32@2x.png:64"
  "icon_128x128.png:128"
  "icon_128x128@2x.png:256"
  "icon_256x256.png:256"
  "icon_256x256@2x.png:512"
  "icon_512x512.png:512"
  "icon_512x512@2x.png:1024"
)

echo "Building ${ICONSET} from $1…"
for entry in "${entries[@]}"; do
  name="${entry%%:*}"
  size="${entry##*:}"
  sips -z "$size" "$size" "$SRC" --out "${ICONSET}/${name}" >/dev/null
  echo "✓ ${name} (${size}x${size})"
done

iconutil -c icns "$ICONSET" -o "${BASE}.icns"
echo "✔ Created ${BASE}.icns"
