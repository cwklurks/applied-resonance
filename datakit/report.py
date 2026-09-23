"""Markdown ingest report for field-recorded datakit clips."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from engine.paths import REPO_ROOT

DEFAULT_DATA_DIR = REPO_ROOT / "datakit" / "data"


@dataclass(frozen=True)
class ClipRow:
    stem: str
    machine_type: str
    condition: str
    site_tag: str
    duration_s: float
    clipped: bool
    contains_speech: bool


def build_report(data_dir: Path = DEFAULT_DATA_DIR) -> str:
    rows = list(load_rows(data_dir))
    total_minutes = sum(row.duration_s for row in rows) / 60.0
    short = [row for row in rows if row.duration_s < 3.0]
    clipped = [row for row in rows if row.clipped]
    speech = sum(1 for row in rows if row.contains_speech)

    lines: list[str] = [
        "# Applied Resonance Datakit Ingest Report",
        "",
        f"- Data dir: `{data_dir}`",
        f"- Clips: {len(rows)}",
        f"- Total minutes: {total_minutes:.2f}",
        f"- Contains speech: {speech}",
        f"- Under 3 s: {len(short)}",
        f"- Clipped: {len(clipped)}",
        "",
        "## Machine Type",
        "",
        count_table(rows, "machine_type"),
        "",
        "## Condition",
        "",
        count_table(rows, "condition"),
        "",
        "## Site/Machine Tag",
        "",
        count_table(rows, "site_tag"),
        "",
        "## Flags",
        "",
        flag_table(short, "Clips under 3 s"),
        "",
        flag_table(clipped, "Clipped clips"),
        "",
    ]
    return "\n".join(lines)


def load_rows(data_dir: Path) -> Iterable[ClipRow]:
    for json_path in sorted(data_dir.glob("*.json")):
        try:
            meta = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            continue

        wav_path = json_path.with_suffix(".wav")
        duration_s, peak_abs = audio_stats(wav_path, meta)
        clipped = bool(meta.get("clipped", False)) or peak_abs >= 0.999
        yield ClipRow(
            stem=json_path.stem,
            machine_type=nonempty(meta.get("machine_type"), "unknown"),
            condition=condition_label(meta),
            site_tag=nonempty(meta.get("site_tag"), "unlabeled"),
            duration_s=duration_s,
            clipped=clipped,
            contains_speech=bool(meta.get("contains_speech", False)),
        )


def audio_stats(wav_path: Path, meta: dict) -> tuple[float, float]:
    sr = int(meta.get("sr") or 0)
    n_samples = int(meta.get("n_samples") or 0)
    if wav_path.exists():
        audio, actual_sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        peak_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
        duration_s = float(audio.shape[0] / actual_sr) if actual_sr else 0.0
        return duration_s, peak_abs
    if sr > 0 and n_samples > 0:
        return float(n_samples / sr), float(meta.get("peak_abs") or 0.0)
    return float(meta.get("duration_s") or 0.0), float(meta.get("peak_abs") or 0.0)


def count_table(rows: list[ClipRow], field: str) -> str:
    counts: Counter[str] = Counter(getattr(row, field) for row in rows)
    durations: dict[str, float] = defaultdict(float)
    for row in rows:
        durations[getattr(row, field)] += row.duration_s
    if not counts:
        return "_No clips._"
    lines = ["| Value | Clips | Minutes |", "|---|---:|---:|"]
    for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {escape_cell(value)} | {count} | {durations[value] / 60.0:.2f} |")
    return "\n".join(lines)


def flag_table(rows: list[ClipRow], title: str) -> str:
    if not rows:
        return f"### {title}\n\n_None._"
    lines = [
        f"### {title}",
        "",
        "| Clip | Machine | Condition | Site tag | Seconds |",
        "|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_cell(row.stem),
                    escape_cell(row.machine_type),
                    escape_cell(row.condition),
                    escape_cell(row.site_tag),
                    f"{row.duration_s:.2f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def condition_label(meta: dict) -> str:
    condition = str(meta.get("condition") or "").strip()
    if condition:
        return condition
    suspected_fault = str(meta.get("suspected_fault") or "").strip()
    return suspected_fault or "unknown"


def nonempty(value: object, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, help="write markdown to this path")
    args = parser.parse_args(argv)

    report = build_report(args.data_dir)
    if args.out:
        args.out.write_text(report)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
