from pathlib import Path
import re
import textwrap

from PIL import Image, ImageDraw, ImageFont


W, H = 1920, 1080
ASSETS = Path("/Users/connork/code/earsight/demo_assets")
OUT = ASSETS / "g2_clarified_overlays"
OUT.mkdir(exist_ok=True)

FONT_DIR = ASSETS / "fonts"
REGULAR = FONT_DIR / "Geist-Regular.ttf"
MEDIUM = FONT_DIR / "Geist-Medium.ttf"
BOLD = FONT_DIR / "Geist-Bold.ttf"

INK = (246, 247, 250, 255)
MUTED = (170, 179, 193, 255)
GREEN = (50, 255, 82, 255)
YELLOW = (255, 213, 46, 255)
RED = (255, 82, 82, 255)
PANEL = (9, 12, 18, 232)
PANEL_SOFT = (9, 12, 18, 210)
STROKE = (62, 72, 88, 255)


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    return image, ImageDraw.Draw(image)


def rounded_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill=PANEL,
    outline=STROKE,
    radius=24,
    width=2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def phone_header(draw: ImageDraw.ImageDraw) -> None:
    # The companion screen occupies x=296..596 and starts at y=78.
    # This header shares those exact edges so it reads as one component.
    box = (296, 26, 596, 78)
    draw.rounded_rectangle(
        box,
        radius=12,
        fill=(9, 12, 18, 244),
        outline=(92, 105, 125, 255),
        width=2,
    )
    draw.text((314, 36), "PHONE COMPANION", font=font(MEDIUM, 15), fill=INK)
    draw.text((314, 56), "mirrors the G2 state", font=font(REGULAR, 12), fill=MUTED)


def write_stage(
    filename: str,
    *,
    number: str,
    eyebrow: str,
    title: str,
    subtitle: str,
    accent=GREEN,
    badge: str | None = None,
) -> None:
    image, draw = canvas()
    phone_header(draw)

    x1, y1, x2, y2 = 1210, 78, 1824, 296
    rounded_panel(draw, (x1, y1, x2, y2))
    draw.rounded_rectangle((x1, y1, x1 + 8, y2), radius=4, fill=accent)

    draw.text((1244, 105), f"{number}  {eyebrow.upper()}", font=font(MEDIUM, 16), fill=accent)
    draw.text((1244, 141), title, font=font(BOLD, 34), fill=INK)
    draw.text((1244, 194), subtitle, font=font(REGULAR, 20), fill=MUTED)

    if badge:
        badge_box = (1244, 238, 1244 + 300, 278)
        draw.rounded_rectangle(badge_box, radius=20, fill=(24, 46, 31, 255))
        draw.text((1263, 248), badge, font=font(MEDIUM, 15), fill=GREEN)

    condensed = (1210, 312, 1824, 354)
    draw.rounded_rectangle(condensed, radius=18, fill=PANEL_SOFT, outline=STROKE, width=1)
    draw.text(
        (1232, 324),
        "TIME CONDENSED  ·  SAME PUMP  ·  SAME VOLUME",
        font=font(MEDIUM, 14),
        fill=MUTED,
    )
    image.save(OUT / filename)


image, draw = canvas()
rounded_panel(draw, (86, 92, 780, 264), fill=PANEL_SOFT)
draw.text((120, 122), "WHY THE G2", font=font(MEDIUM, 17), fill=GREEN)
draw.text((120, 158), "Hands-free, eyes-up machine monitoring", font=font(BOLD, 30), fill=INK)
draw.text(
    (120, 210),
    "The lens shows the warning. A temple tap saves evidence.",
    font=font(REGULAR, 19),
    fill=MUTED,
)
image.save(OUT / "00-why-g2.png")

write_stage(
    "01-learn-normal.png",
    number="01",
    eyebrow="Learn normal",
    title="Build a baseline",
    subtitle="G2 microphone listens to the normal pump",
    accent=YELLOW,
)
write_stage(
    "02-change-sound.png",
    number="02",
    eyebrow="Same machine",
    title="The sound changes",
    subtitle="An abnormal recording replaces the normal one",
    accent=YELLOW,
)
write_stage(
    "02b-alert-on-glasses.png",
    number="02",
    eyebrow="Persistent change",
    title="Alert reaches the G2 lens",
    subtitle="The wearer sees it without opening a phone",
    accent=RED,
)
write_stage(
    "03-save-evidence.png",
    number="03",
    eyebrow="Temple tap",
    title="Save the last 10 seconds",
    subtitle="One gesture logs the evidence window",
    accent=GREEN,
    badge="✓  EVIDENCE SAVED",
)
write_stage(
    "04-recover.png",
    number="04",
    eyebrow="Normal returns",
    title="Re-arm to listening",
    subtitle="The detector recovers instead of staying alarmed",
    accent=GREEN,
)

image, draw = canvas()
rounded_panel(draw, (1190, 72, 1840, 840), fill=(9, 12, 18, 248))
draw.text((1232, 112), "WHY THE G2", font=font(MEDIUM, 18), fill=GREEN)
draw.text((1232, 153), "The glasses are the interface", font=font(BOLD, 34), fill=INK)
draw.text(
    (1232, 211),
    "The worker keeps both hands free.",
    font=font(REGULAR, 19),
    fill=MUTED,
)
draw.text(
    (1232, 239),
    "The warning stays in sight.",
    font=font(REGULAR, 19),
    fill=MUTED,
)

flow_y = 300
flow_items = [
    ("G2 MIC", "captures the machine sound"),
    ("PHONE", "streams audio to the detector"),
    ("MODEL", "compares it with the baseline"),
    ("G2 LENS", "shows a two-line warning"),
]
for index, (label, detail) in enumerate(flow_items, start=1):
    cy = flow_y + (index - 1) * 108
    draw.ellipse((1232, cy, 1272, cy + 40), fill=(24, 46, 31, 255))
    draw.text((1245, cy + 10), str(index), font=font(BOLD, 16), fill=GREEN)
    draw.text((1294, cy - 1), label, font=font(MEDIUM, 18), fill=INK)
    draw.text((1294, cy + 29), detail, font=font(REGULAR, 17), fill=MUTED)
    if index < len(flow_items):
        draw.line((1252, cy + 47, 1252, cy + 94), fill=(75, 92, 111, 255), width=2)

draw.rounded_rectangle((1232, 744, 1798, 802), radius=18, fill=(24, 46, 31, 255))
draw.text(
    (1254, 762),
    "TEMPLE TAP  →  save the previous 10 seconds",
    font=font(MEDIUM, 17),
    fill=GREEN,
)
draw.text(
    (1232, 814),
    "The phone view is a companion mirror, not the optical display.",
    font=font(REGULAR, 14),
    fill=MUTED,
)
image.save(OUT / "05-g2-loop.png")

image, draw = canvas()
phone_header(draw)
image.save(OUT / "06-phone-companion-header.png")

CAPTION_OUT = OUT / "captions"
CAPTION_OUT.mkdir(exist_ok=True)
SRT = ASSETS / "Applied Resonance - Hack the North Application Cut - Final 89s.srt"


def timestamp_to_seconds(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, milliseconds = rest.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000
    )


timing_lines: list[str] = []
for block in re.split(r"\n\s*\n", SRT.read_text().strip()):
    lines = block.splitlines()
    cue = int(lines[0])
    start_text, end_text = lines[1].split(" --> ")
    raw_text = " ".join(lines[2:])
    wrapped = textwrap.wrap(raw_text, width=58, break_long_words=False)

    image, draw = canvas()
    caption_font = font(MEDIUM, 30)
    spacing = 8
    bounds = draw.multiline_textbbox(
        (0, 0),
        "\n".join(wrapped),
        font=caption_font,
        spacing=spacing,
        align="center",
    )
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    card_width = min(1390, text_width + 64)
    card_height = text_height + 38
    left = (W - card_width) // 2
    top = H - 48 - card_height
    draw.rounded_rectangle(
        (left, top, left + card_width, top + card_height),
        radius=18,
        fill=(7, 9, 13, 218),
        outline=(74, 82, 96, 180),
        width=1,
    )
    draw.multiline_text(
        (W // 2, top + 18),
        "\n".join(wrapped),
        font=caption_font,
        fill=INK,
        spacing=spacing,
        anchor="ma",
        align="center",
    )

    filename = f"caption_{cue:02d}.png"
    image.save(CAPTION_OUT / filename)
    timing_lines.append(
        f"{cue}\t{timestamp_to_seconds(start_text):.3f}\t"
        f"{timestamp_to_seconds(end_text):.3f}\t{filename}"
    )

(CAPTION_OUT / "timings.tsv").write_text("\n".join(timing_lines) + "\n")

print(OUT)
