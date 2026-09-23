#!/bin/bash
set -euo pipefail

FFMPEG="/opt/homebrew/Cellar/ffmpeg/8.1.1/bin/ffmpeg"
ASSETS="/Users/connork/code/earsight/demo_assets"
EXPLAINERS="$ASSETS/explainer_overlays"
CAPTIONS="/private/tmp/applied_resonance_caption_cards"
SOURCE="$ASSETS/Applied Resonance - Hack the North Application Cut - Final 89s.mp4"
OUTPUT="$ASSETS/Applied Resonance - Hack the North Application Cut - Explained Final 89s.mp4"

args=(-hide_banner -y -i "$SOURCE")
filter=""
previous="0:v"
input_index=1

add_overlay() {
  local filename="$1"
  local start="$2"
  local end="$3"
  local name="$4"
  args+=(-loop 1 -framerate 30 -i "$filename")
  filter+="[${previous}][${input_index}:v]overlay=0:0:enable='between(t,${start},${end})'[${name}];"
  previous="$name"
  input_index=$((input_index + 1))
}

add_overlay "$ASSETS/pump-test-card.png" 9.660 15.640 overview
add_overlay "$EXPLAINERS/01-learn-normal.png" 15.640 22.440 stage01
add_overlay "$EXPLAINERS/02-change-sound.png" 22.440 30.880 stage02
add_overlay "$EXPLAINERS/03-save-evidence.png" 30.880 36.280 stage03
add_overlay "$EXPLAINERS/04-recover.png" 36.280 47.490 stage04
add_overlay "$ASSETS/dcase-results-card.png" 47.490 70.640 benchmark
add_overlay "$EXPLAINERS/05-working-loop.png" 75.560 81.880 pipeline

while IFS=$'\t' read -r cue start end filename; do
  args+=(-loop 1 -framerate 30 -i "$CAPTIONS/$filename")
  next="caption${cue}"
  filter+="[${previous}][${input_index}:v]overlay=(W-w)/2:H-h-48:enable='between(t,${start},${end})'[${next}];"
  previous="$next"
  input_index=$((input_index + 1))
done < "$CAPTIONS/timings.tsv"

add_overlay "$ASSETS/applied-resonance-white-end-card.png" 88.000 88.600 endcard

filter="${filter%;}"

"$FFMPEG" "${args[@]}" \
  -filter_complex "$filter" \
  -map "[$previous]" \
  -map 0:a \
  -t 88.60 \
  -c:v libx264 \
  -preset medium \
  -crf 18 \
  -pix_fmt yuv420p \
  -c:a copy \
  -movflags +faststart \
  "$OUTPUT"
