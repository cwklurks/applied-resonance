import json

import numpy as np
import soundfile as sf

from datakit.report import build_report


SR = 16000


def _write_clip(root, stem, audio, meta):
    sf.write(str(root / f"{stem}.wav"), audio, SR, subtype="PCM_16")
    sidecar = {
        "machine_type": meta.get("machine_type", "fan"),
        "condition": meta.get("condition", "sounds normal"),
        "site_tag": meta.get("site_tag", "site-a"),
        "contains_speech": meta.get("contains_speech", False),
        "sr": SR,
        "n_samples": len(audio),
        **meta,
    }
    (root / f"{stem}.json").write_text(json.dumps(sidecar))


def test_report_groups_counts_minutes_and_flags(tmp_path):
    _write_clip(
        tmp_path,
        "normal_long",
        np.zeros(4 * SR, dtype=np.float32),
        {
            "machine_type": "fan",
            "condition": "sounds normal",
            "site_tag": "roof-fan-1",
        },
    )
    _write_clip(
        tmp_path,
        "short_clipped",
        np.ones(2 * SR, dtype=np.float32),
        {
            "machine_type": "pump",
            "condition": "confirmed issue",
            "site_tag": "pump-2",
            "contains_speech": True,
        },
    )

    report = build_report(tmp_path)

    assert "# Applied Resonance Datakit Ingest Report" in report
    assert "- Clips: 2" in report
    assert "- Contains speech: 1" in report
    assert "- Under 3 s: 1" in report
    assert "- Clipped: 1" in report
    assert "| fan | 1 |" in report
    assert "| pump | 1 |" in report
    assert "| confirmed issue | 1 |" in report
    assert "short_clipped" in report
