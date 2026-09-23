#!/bin/bash
set -euo pipefail

FFMPEG="/opt/homebrew/Cellar/ffmpeg/8.1.1/bin/ffmpeg"
ROOT="/Users/connork/code/earsight"
ASSETS="$ROOT/demo_assets"
OVERLAYS="$ASSETS/g2_clarified_overlays"
SOURCE="$ASSETS/Applied Resonance - Hack the North Application Cut - Final 89s.mp4"
EXPERIMENT="$ASSETS/Applied Resonance - Hack the North Demo - Pre-Caption Side HUD.mp4"
CAMERA="$ROOT/demo_backup_pre_caption_2026-07-27/resolve_media/Applied Resonance HTN Demo/applied resonance demo.mov"
SRT="$ASSETS/Applied Resonance - Hack the North Application Cut - Final 89s.srt"
OUTPUT="$ASSETS/Applied Resonance - Hack the North Application Cut - G2 Clarified Final 88.6s.mp4"
CAPTIONS="$OVERLAYS/captions"

args=(
  -hide_banner -y
  -i "$SOURCE"
  -i "$EXPERIMENT"
  -i "$CAMERA"
  -loop 1 -framerate 30 -i "$ASSETS/dcase-results-card.png"
  -loop 1 -framerate 30 -i "$ASSETS/applied-resonance-white-end-card.png"
  -loop 1 -framerate 30 -i "$OVERLAYS/00-why-g2.png"
  -loop 1 -framerate 30 -i "$OVERLAYS/01-learn-normal.png"
  -loop 1 -framerate 30 -i "$OVERLAYS/02-change-sound.png"
  -loop 1 -framerate 30 -i "$OVERLAYS/02b-alert-on-glasses.png"
  -loop 1 -framerate 30 -i "$OVERLAYS/03-save-evidence.png"
  -loop 1 -framerate 30 -i "$OVERLAYS/04-recover.png"
  -loop 1 -framerate 30 -i "$OVERLAYS/05-g2-loop.png"
  -loop 1 -framerate 30 -i "$OVERLAYS/06-phone-companion-header.png"
)

filter="
    [0:v]trim=start=0:end=10.760,setpts=PTS-STARTPTS,fps=30,scale=1920:1080,setsar=1[v0];
    [1:v]trim=start=29.000:end=40.680,setpts=PTS-STARTPTS,fps=30,scale=1920:1080,setsar=1[v1];
    [1:v]trim=start=59.000:end=63.720,setpts=PTS-STARTPTS,fps=30,scale=1920:1080,setsar=1[v2];
    [1:v]trim=start=88.000:end=91.720,setpts=PTS-STARTPTS,fps=30,scale=1920:1080,setsar=1[v3];
    [1:v]trim=start=94.000:end=99.400,setpts=PTS-STARTPTS,fps=30,scale=1920:1080,setsar=1[v4];
    [1:v]trim=start=106.000:end=113.000,
      setpts=(PTS-STARTPTS)*1.6343,fps=30,scale=1920:1080,setsar=1[v5];
    [3:v]trim=start=0:end=22.920,setpts=PTS-STARTPTS,fps=30,scale=1920:1080,setsar=1[v6];
    [2:v]trim=start=68.000:end=81.040,setpts=PTS-STARTPTS,fps=30,
      crop=800:720:160:0,scale=1200:1080,
      pad=1920:1080:0:0:color=0x090c12,setsar=1[v7];
    [0:v]trim=start=83.680:end=88.000,setpts=PTS-STARTPTS,fps=30,scale=1920:1080,setsar=1[v8];
    [4:v]trim=start=0:end=0.618,setpts=PTS-STARTPTS,fps=30,scale=1920:1080,setsar=1[v9];
    [v0][v1][v2][v3][v4][v5][v6][v7][v8][v9]
      concat=n=10:v=1:a=0[base];
    [base][5:v]overlay=0:0:enable='between(t,6.250,10.760)'[why];
    [why][6:v]overlay=0:0:enable='between(t,10.760,22.440)'[stage1];
    [stage1][7:v]overlay=0:0:enable='between(t,22.440,27.160)'[stage2];
    [stage2][8:v]overlay=0:0:enable='between(t,27.160,30.880)'[alert];
    [alert][9:v]overlay=0:0:enable='between(t,30.880,36.280)'[save];
    [save][10:v]overlay=0:0:enable='between(t,36.280,47.720)'[recover];
    [recover][11:v]overlay=0:0:enable='between(t,70.640,83.680)'[g2loop];
    [g2loop][12:v]overlay=0:0:enable='between(t,83.680,88.000)'[phoneclose]"

previous="phoneclose"
input_index=13
caption_chain=""
caption_count=0
while IFS=$'\t' read -r cue start end filename; do
  args+=(-loop 1 -framerate 30 -i "$CAPTIONS/$filename")
  duration=$(awk -v start="$start" -v end="$end" 'BEGIN { printf "%.3f", end - start }')
  filter+=";[${input_index}:v]trim=start=0:end=${duration},setpts=PTS-STARTPTS,fps=30[caption${cue}]"
  caption_chain+="[caption${cue}]"
  caption_count=$((caption_count + 1))
  input_index=$((input_index + 1))
done < "$CAPTIONS/timings.tsv"

filter+=";${caption_chain}concat=n=${caption_count}:v=1:a=0[captionstream]"
filter+=";[${previous}][captionstream]overlay=0:0:eof_action=pass[captioned]"
previous="captioned"
filter+=";[0:a]atrim=start=0:end=88.618,asetpts=PTS-STARTPTS[audio]"

"$FFMPEG" "${args[@]}" \
  -filter_complex "$filter" \
  -map "[$previous]" \
  -map "[audio]" \
  -t 88.618 \
  -c:v libx264 \
  -preset medium \
  -crf 18 \
  -pix_fmt yuv420p \
  -c:a aac \
  -b:a 192k \
  -movflags +faststart \
  "$OUTPUT"

printf '%s\n' "$OUTPUT"
