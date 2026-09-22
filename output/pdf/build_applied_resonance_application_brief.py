#!/usr/bin/env python3
"""Build the two-page Applied Resonance Hack the North application brief."""

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "Applied_Resonance_Application_Brief_Connor_Klann.pdf"
BRAND_LOGO = ROOT / "brand images" / "applied-resonance-horizontal.png"

PAGE_W, PAGE_H = letter
MARGIN = 46

CREAM = HexColor("#F7F4ED")
PAPER = HexColor("#FFFDF9")
INK = HexColor("#0B1220")
SLATE = HexColor("#596474")
MUTED = HexColor("#89919D")
LINE = HexColor("#D9D3C8")
SHELL = HexColor("#EAE5DB")
MINT = HexColor("#67E4C2")
MINT_DARK = HexColor("#126455")
MINT_WASH = HexColor("#DFF7EF")
ORANGE = HexColor("#F0A15A")
ORANGE_WASH = HexColor("#FCE8D5")
NAVY = HexColor("#07110E")
WHITE = HexColor("#FFFFFF")


def register_fonts() -> None:
    font_dir = ROOT / "brand images" / "fonts"
    regular = font_dir / "Geist-Regular.ttf"
    medium = font_dir / "Geist-Medium.ttf"
    bold = font_dir / "Geist-Bold.ttf"
    pdfmetrics.registerFont(TTFont("Brief-Regular", regular))
    pdfmetrics.registerFont(TTFont("Brief-Medium", medium))
    pdfmetrics.registerFont(TTFont("Brief-Bold", bold))


def style(
    name: str,
    size: float,
    leading: float,
    color=INK,
    font: str = "Brief-Regular",
    alignment: int = TA_LEFT,
    space_after: float = 0,
) -> ParagraphStyle:
    return ParagraphStyle(
        name=name,
        fontName=font,
        fontSize=size,
        leading=leading,
        textColor=color,
        alignment=alignment,
        spaceAfter=space_after,
        allowWidows=0,
        allowOrphans=0,
    )


BODY = style("Body", 9.2, 13.6, SLATE)
BODY_DARK = style("BodyDark", 9.2, 13.6, INK)
BODY_COMPACT = style("BodyCompact", 8.4, 12.3, SLATE)
SMALL = style("Small", 7.25, 10.3, SLATE)
SMALL_DARK = style("SmallDark", 7.25, 10.3, INK)
LABEL = style("Label", 6.9, 8.4, SLATE, "Brief-Medium")
LABEL_MINT = style("LabelMint", 6.9, 8.4, MINT_DARK, "Brief-Medium")
LABEL_ORANGE = style("LabelOrange", 6.9, 8.4, HexColor("#8B4B18"), "Brief-Medium")


def draw_paragraph(c: canvas.Canvas, text: str, x: float, y_top: float, width: float, pstyle: ParagraphStyle) -> float:
    p = Paragraph(text, pstyle)
    _, height = p.wrap(width, PAGE_H)
    p.drawOn(c, x, y_top - height)
    return height


def draw_label(c: canvas.Canvas, text: str, x: float, y: float, color=SLATE) -> None:
    c.saveState()
    c.setFillColor(color)
    label = c.beginText(x, y)
    label.setFont("Brief-Medium", 6.7)
    label.setCharSpace(1.15)
    label.textLine(text.upper())
    c.drawText(label)
    c.restoreState()


def draw_logo(c: canvas.Canvas, x: float, y: float) -> None:
    width = 171
    height = width * 173 / 1113
    c.drawImage(
        ImageReader(BRAND_LOGO),
        x,
        y - height / 2,
        width=width,
        height=height,
        preserveAspectRatio=True,
        mask="auto",
    )


def draw_header(c: canvas.Canvas, page_label: str) -> None:
    draw_logo(c, MARGIN, PAGE_H - 52)
    c.setFillColor(SLATE)
    c.setFont("Brief-Medium", 6.8)
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 54, page_label.upper())
    c.setStrokeColor(LINE)
    c.setLineWidth(0.55)
    c.line(MARGIN, PAGE_H - 74, PAGE_W - MARGIN, PAGE_H - 74)


def draw_footer(c: canvas.Canvas, page_number: int) -> None:
    c.setStrokeColor(LINE)
    c.setLineWidth(0.55)
    c.line(MARGIN, 35, PAGE_W - MARGIN, 35)
    c.setFillColor(SLATE)
    c.setFont("Brief-Medium", 6.4)
    c.drawString(MARGIN, 21, "Connor Klann | Vancouver, BC | connork.com | github.com/cwklurks")
    c.drawRightString(PAGE_W - MARGIN, 21, f"{page_number} / 2")
    c.linkURL("https://www.connork.com/", (MARGIN + 112, 16, MARGIN + 164, 29), relative=0)
    c.linkURL("https://github.com/cwklurks", (MARGIN + 171, 16, MARGIN + 265, 29), relative=0)


def draw_glasses(c: canvas.Canvas, x: float, y: float, scale: float = 1.0) -> None:
    c.setLineWidth(2.2 * scale)
    c.setStrokeColor(INK)
    c.roundRect(x, y, 54 * scale, 29 * scale, 9 * scale, stroke=1, fill=0)
    c.roundRect(x + 68 * scale, y, 54 * scale, 29 * scale, 9 * scale, stroke=1, fill=0)
    c.line(x + 54 * scale, y + 17 * scale, x + 68 * scale, y + 17 * scale)
    c.line(x - 13 * scale, y + 23 * scale, x, y + 20 * scale)
    c.line(x + 122 * scale, y + 20 * scale, x + 135 * scale, y + 23 * scale)
    c.setStrokeColor(MINT_DARK)
    c.setLineWidth(1.3 * scale)
    c.line(x + 77 * scale, y + 7 * scale, x + 112 * scale, y + 7 * scale)
    c.line(x + 77 * scale, y + 12 * scale, x + 102 * scale, y + 12 * scale)


def draw_architecture(c: canvas.Canvas, x: float, y: float, width: float) -> None:
    labels = [
        ("LISTEN", "G2 mic or audio"),
        ("LEARN", "30 s baseline"),
        ("COMPARE", "3 s windows"),
        ("SCORE", "PANNs + kNN"),
        ("ALERT", "two-line HUD"),
    ]
    height = 64
    c.setFillColor(SHELL)
    c.roundRect(x, y, width, height, 13, stroke=0, fill=1)
    c.setFillColor(PAPER)
    c.roundRect(x + 3, y + 3, width - 6, height - 6, 10, stroke=0, fill=1)
    node_w = (width - 6) / 5
    for index, (title, detail) in enumerate(labels):
        nx = x + 3 + index * node_w
        if index:
            c.setStrokeColor(LINE)
            c.setLineWidth(0.5)
            c.line(nx, y + 13, nx, y + height - 13)
        c.setFillColor(MINT_DARK if index in (0, 4) else MUTED)
        c.setFont("Brief-Medium", 5.8)
        c.drawString(nx + 11, y + 44, f"0{index + 1}")
        c.setFillColor(MINT_DARK if index in (0, 4) else INK)
        c.setFont("Brief-Medium", 7.4)
        c.drawString(nx + 11, y + 29, title)
        draw_paragraph(c, detail, nx + 11, y + 21, node_w - 22, SMALL)


def draw_metric_band(c: canvas.Canvas, x: float, y: float, width: float) -> None:
    height = 94
    inset = 3
    scale_w = 152
    c.setFillColor(SHELL)
    c.roundRect(x, y, width, height, 14, stroke=0, fill=1)
    c.setFillColor(PAPER)
    c.roundRect(x + inset, y + inset, width - 2 * inset, height - 2 * inset, 11, stroke=0, fill=1)

    c.setFillColor(NAVY)
    c.roundRect(x + inset, y + inset, scale_w, height - 2 * inset, 11, stroke=0, fill=1)
    c.rect(x + scale_w - 7, y + inset, 10, height - 2 * inset, stroke=0, fill=1)
    draw_label(c, "Evaluation scale", x + 17, y + 69, MINT)
    c.setFillColor(WHITE)
    c.setFont("Brief-Bold", 24)
    c.drawString(x + 17, y + 38, "9,755")
    draw_paragraph(c, "fan and pump recordings", x + 17, y + 29, scale_w - 30, style("MetricLight", 7, 9, HexColor("#AFC6BE")))

    data_x = x + scale_w + 18
    data_w = width - scale_w - 34
    column_w = data_w / 2
    for index, (title, value, reference, color) in enumerate(
        [
            ("FAN AUC", "0.693", "DCASE 0.658  |  +3.5 pts", ORANGE),
            ("PUMP AUC", "0.880", "DCASE 0.729  |  +15.1 pts", MINT),
        ]
    ):
        mx = data_x + index * column_w
        if index:
            c.setStrokeColor(LINE)
            c.setLineWidth(0.5)
            c.line(mx - 10, y + 18, mx - 10, y + height - 18)
        c.setFillColor(color)
        c.circle(mx + 3, y + 73, 2.4, stroke=0, fill=1)
        c.setFillColor(SLATE)
        c.setFont("Brief-Medium", 6.6)
        c.drawString(mx + 11, y + 70, title)
        c.setFillColor(INK)
        c.setFont("Brief-Bold", 23)
        c.drawString(mx, y + 39, value)
        c.setFillColor(SLATE)
        c.setFont("Brief-Regular", 6.8)
        c.drawString(mx, y + 23, reference)


def draw_bullet_list(c: canvas.Canvas, items: list[str], x: float, y_top: float, width: float, color=INK) -> None:
    y = y_top
    for item in items:
        c.setFillColor(color)
        c.circle(x + 2.5, y - 4.4, 1.8, stroke=0, fill=1)
        height = draw_paragraph(c, item, x + 12, y, width - 12, SMALL_DARK)
        y -= max(height, 12.5)


def page_one(c: canvas.Canvas) -> None:
    c.setFillColor(CREAM)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    draw_header(c, "Application brief / July 2026")

    c.setFillColor(INK)
    c.setFont("Brief-Bold", 36)
    c.drawString(MARGIN, 665, "Smart glasses that listen")
    c.drawString(MARGIN, 621, "for machine faults.")
    c.setFillColor(SLATE)
    c.setFont("Brief-Regular", 10.8)
    c.drawString(MARGIN, 594, "A self-funded hardware pivot that became a benchmarked audio ML system.")

    draw_label(c, "Why I built it", MARGIN, 552)
    story = (
        "About six months ago, my boxing coach came to the gym wearing Ray-Ban Meta glasses. "
        "Seeing someone I knew use smart glasses made the technology feel real. I wanted to build "
        "for that future, so I completed coding bounties and programming work for my dad, saved the "
        "money, and bought Even Realities G2 glasses myself. Applied Resonance grew from one question: could "
        "smart glasses give a technician another way to notice a machine problem before it becomes "
        "obvious?"
    )
    c.setStrokeColor(MINT)
    c.setLineWidth(2.4)
    c.line(MARGIN, 529, MARGIN, 463)
    draw_paragraph(c, story, MARGIN + 13, 535, 313, BODY)

    card_x, card_y, card_w, card_h = 390, 459, 176, 103
    c.setFillColor(HexColor("#CFEDE3"))
    c.roundRect(card_x, card_y, card_w, card_h, 16, stroke=0, fill=1)
    c.setFillColor(MINT_WASH)
    c.roundRect(card_x + 3, card_y + 3, card_w - 6, card_h - 6, 13, stroke=0, fill=1)
    draw_label(c, "The idea", card_x + 17, card_y + 78, MINT_DARK)
    draw_glasses(c, card_x + 29, card_y + 35, 0.72)
    draw_paragraph(
        c,
        "Learn a healthy sound. Flag the change. Explain it in two lines.",
        card_x + 17,
        card_y + 30,
        card_w - 34,
        SMALL_DARK,
    )

    draw_label(c, "How Applied Resonance works", MARGIN, 423)
    draw_architecture(c, MARGIN, 346, PAGE_W - 2 * MARGIN)
    c.setFillColor(SLATE)
    c.setFont("Brief-Regular", 6.5)
    c.drawRightString(PAGE_W - MARGIN, 335, "Current implementation: local scoring engine, no cloud inference required")

    c.setFillColor(HexColor("#CBD1CA"))
    c.roundRect(MARGIN, 228, PAGE_W - 2 * MARGIN, 87, 18, stroke=0, fill=1)
    c.setFillColor(NAVY)
    c.roundRect(MARGIN + 3, 232, PAGE_W - 2 * MARGIN - 6, 79, 15, stroke=0, fill=1)
    c.setFillColor(MINT)
    c.circle(MARGIN + 28, 272, 5, stroke=0, fill=1)
    c.setFont("Brief-Medium", 15.5)
    c.drawString(MARGIN + 48, 273, "Bearing-like anomaly | 98%")
    c.setFont("Brief-Regular", 10)
    c.drawString(MARGIN + 48, 250, "high-band energy rising | tap to log")
    c.setFillColor(HexColor("#9DB8AF"))
    c.setFont("Brief-Regular", 7)
    c.drawRightString(PAGE_W - MARGIN - 18, 246, "TWO-LINE G2 HUD CONTRACT")

    draw_label(c, "What makes it different", MARGIN, 196)
    innovation = (
        "Most monitoring systems attach a sensor to one machine. Applied Resonance explores a different model: "
        "the microphone moves with the technician, while the display stays hands-free. The same pair of "
        "glasses can move between machines, surface an anomaly, and save evidence for later review."
    )
    c.setStrokeColor(MINT)
    c.setLineWidth(2.4)
    c.line(MARGIN, 174, MARGIN, 132)
    draw_paragraph(c, innovation, MARGIN + 13, 179, 329, BODY)

    c.setFillColor(SHELL)
    c.roundRect(405, 128, 161, 68, 14, stroke=0, fill=1)
    c.setFillColor(PAPER)
    c.roundRect(408, 131, 155, 62, 11, stroke=0, fill=1)
    draw_label(c, "Built with", 423, 175)
    draw_paragraph(c, "Python / PyTorch<br/>TypeScript / Even Hub<br/>PANNs / kNN / ONNX", 423, 161, 126, SMALL_DARK)

    draw_footer(c, 1)
    c.showPage()


def page_two(c: canvas.Canvas) -> None:
    c.setFillColor(CREAM)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    draw_header(c, "Evidence, limits, and next gate")

    c.setFillColor(INK)
    c.setFont("Brief-Bold", 29)
    c.drawString(MARGIN, 669, "What I proved, and what")
    c.drawString(MARGIN, 634, "I still need to prove.")
    c.setFillColor(SLATE)
    c.setFont("Brief-Regular", 10.2)
    c.drawString(MARGIN, 609, "The benchmark is proven. The real G2 microphone is the next gate.")

    draw_metric_band(c, MARGIN, 492, PAGE_W - 2 * MARGIN)

    dcase = (
        "DCASE is an international research challenge for detecting and classifying sounds. Its 2020 "
        "Task 2 autoencoder is the official reference model for machine anomaly detection. Applied Resonance was "
        "fit only on healthy machine clips, then evaluated on held-out healthy and faulty clips. AUC is a "
        "ranking metric where 0.5 is chance and 1.0 is perfect."
    )
    draw_paragraph(c, dcase, MARGIN, 473, PAGE_W - 2 * MARGIN, SMALL)

    col_gap = 14
    left_w = 286
    right_w = PAGE_W - 2 * MARGIN - col_gap - left_w
    right_x = MARGIN + left_w + col_gap
    box_y, box_h = 311, 128
    c.setFillColor(SHELL)
    c.roundRect(MARGIN, box_y, left_w, box_h, 15, stroke=0, fill=1)
    c.setFillColor(PAPER)
    c.roundRect(MARGIN + 3, box_y + 3, left_w - 6, box_h - 6, 12, stroke=0, fill=1)
    draw_label(c, "The failure I did not hide", MARGIN + 17, box_y + 101, HexColor("#8B4B18"))
    failure = (
        "A fan-versus-pump classifier reached 1.000 training accuracy, then dropped to 0.577 on unseen "
        "machine IDs. I removed it from the live path instead of presenting the training score. The anomaly "
        "detector uses a separate per-machine design and was unaffected."
    )
    draw_paragraph(c, failure, MARGIN + 17, box_y + 82, left_w - 34, BODY_COMPACT)

    c.setFillColor(HexColor("#CFEDE3"))
    c.roundRect(right_x, box_y, right_w, box_h, 15, stroke=0, fill=1)
    c.setFillColor(MINT_WASH)
    c.roundRect(right_x + 3, box_y + 3, right_w - 6, box_h - 6, 12, stroke=0, fill=1)
    draw_label(c, "How I built it", right_x + 17, box_y + 101, MINT_DARK)
    build = (
        "This was my first Claude Code build with Fable 5. I split it into 22 independently testable tasks, "
        "wrote acceptance criteria, and checked each result. Faster generation made verification more "
        "important, not less."
    )
    draw_paragraph(c, build, right_x + 17, box_y + 82, right_w - 34, BODY_COMPACT)

    status_y, status_h = 153, 132
    status_left_w = 244
    status_right_x = MARGIN + status_left_w + col_gap
    status_right_w = PAGE_W - MARGIN - status_right_x
    c.setFillColor(HexColor("#CFEDE3"))
    c.roundRect(MARGIN, status_y, status_left_w, status_h, 15, stroke=0, fill=1)
    c.setFillColor(MINT_WASH)
    c.roundRect(MARGIN + 3, status_y + 3, status_left_w - 6, status_h - 6, 12, stroke=0, fill=1)
    draw_label(c, "Validated today", MARGIN + 17, status_y + 105, MINT_DARK)
    draw_bullet_list(
        c,
        [
            "Full MIMII fan and pump evaluation",
            "Deterministic double-pass across all reported metrics",
            "Live streaming, evidence, and labeling engine",
            "Even Hub simulator and real-engine integration",
        ],
        MARGIN + 17,
        status_y + 84,
        status_left_w - 34,
        MINT_DARK,
    )

    c.setFillColor(HexColor("#F2D5B7"))
    c.roundRect(status_right_x, status_y, status_right_w, status_h, 15, stroke=0, fill=1)
    c.setFillColor(ORANGE_WASH)
    c.roundRect(status_right_x + 3, status_y + 3, status_right_w - 6, status_h - 6, 12, stroke=0, fill=1)
    draw_label(c, "Next G2 hardware gate", status_right_x + 17, status_y + 105, HexColor("#8B4B18"))
    draw_bullet_list(
        c,
        [
            "Confirm continuous PCM capture from the real glasses",
            "Verify sample rate with a known 1 kHz tone",
            "Re-record the pump test through the G2 microphone",
            "Measure background operation and battery cost",
        ],
        status_right_x + 17,
        status_y + 84,
        status_right_w - 34,
        HexColor("#8B4B18"),
    )

    c.setFillColor(INK)
    c.setFont("Brief-Medium", 11.2)
    c.drawCentredString(PAGE_W / 2, 111, "The part I am proudest of is that I was willing to find out where it did not work.")
    c.setFillColor(SLATE)
    c.setFont("Brief-Regular", 6.9)
    c.drawCentredString(PAGE_W / 2, 92, "Full 63-page build transcript included in the application folder | Source: engine/REPORT_FULL.md")

    draw_footer(c, 2)
    c.showPage()


def build() -> Path:
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=letter, pageCompression=1)
    c.setTitle("Applied Resonance Application Brief - Connor Klann")
    c.setAuthor("Connor Klann")
    c.setSubject("Hack the North 2026 project evidence")
    page_one(c)
    page_two(c)
    c.save()
    return OUTPUT


if __name__ == "__main__":
    print(build())
