"""Pure-unit tests for engine.download (no network).

A synthetic zip64-capable archive is built with the real ``zipfile`` module,
mixing deflated and stored members in the MIMII layout, and exercised through a
local-file Fetcher.
"""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import pytest

from engine.download import (
    FileFetcher,
    HttpFetcher,
    ZipEntry,
    compute_member_spans,
    extract_member,
    extract_run_member,
    group_runs,
    member_target_path,
    parse_central_directory,
    parse_central_directory_with_offset,
    resume_range_header,
    select_subset,
    split_counts,
)


# --- Synthetic archive fixture ------------------------------------------------


def _payload(machine: str, id_dir: str, condition: str, index: int) -> bytes:
    """Deterministic, per-member-unique fake WAV payload (varied length)."""
    head = f"{machine}/{id_dir}/{condition}/{index:08d}".encode()
    # Vary size so stored vs deflated and run-slicing are non-trivial.
    body = bytes((i * 7 + index * 3 + len(condition)) % 256 for i in range(64 + index * 5))
    return head + b"|" + body


def build_synthetic_zip(
    path: Path,
    machine: str = "fan",
    ids: tuple[str, ...] = ("id_00", "id_02", "id_04", "id_06"),
    n_normal: int = 60,
    n_abnormal: int = 20,
) -> dict[str, bytes]:
    """Write a MIMII-shaped zip; return {member_name: uncompressed_payload}.

    Members alternate stored/deflated. Filenames are zero-padded so sorted order
    is well defined.
    """
    contents: dict[str, bytes] = {}
    with zipfile.ZipFile(path, "w") as zf:
        toggle = 0
        for id_dir in ids:
            # A stored directory entry, like the real archive has.
            zf.writestr(f"{machine}/{id_dir}/", b"")
            for condition, count in (("normal", n_normal), ("abnormal", n_abnormal)):
                for index in range(count):
                    name = f"{machine}/{id_dir}/{condition}/{index:08d}.wav"
                    data = _payload(machine, id_dir, condition, index)
                    method = zipfile.ZIP_STORED if toggle % 2 == 0 else zipfile.ZIP_DEFLATED
                    toggle += 1
                    zf.writestr(zipfile.ZipInfo(name), data, compress_type=method)
                    contents[name] = data
    return contents


@pytest.fixture
def synthetic(tmp_path: Path):
    zip_path = tmp_path / "0_dB_fan.zip"
    contents = build_synthetic_zip(zip_path)
    return zip_path, contents


# --- split_counts -------------------------------------------------------------


@pytest.mark.parametrize(
    "n, expected",
    [
        (50, (35, 15)),
        (4, (3, 1)),
        (1, (1, 0)),
        (0, (0, 0)),
        (10, (7, 3)),
        (100, (70, 30)),
    ],
)
def test_split_counts(n, expected):
    assert split_counts(n) == expected


def test_split_counts_rejects_negative():
    with pytest.raises(ValueError):
        split_counts(-1)


# --- Central directory parsing ------------------------------------------------


def test_parse_central_directory_finds_all_wavs(synthetic):
    zip_path, contents = synthetic
    fetcher = FileFetcher(zip_path)
    entries = parse_central_directory(fetcher)
    wav_names = {e.name for e in entries if e.name.endswith(".wav")}
    assert wav_names == set(contents.keys())


def test_parse_central_directory_sizes_match_zipfile(synthetic):
    zip_path, contents = synthetic
    fetcher = FileFetcher(zip_path)
    entries = {e.name: e for e in parse_central_directory(fetcher)}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".wav"):
                continue
            entry = entries[info.filename]
            assert entry.uncomp_size == info.file_size
            assert entry.comp_size == info.compress_size
            assert entry.method == info.compress_type


def test_parse_central_directory_offset_is_consistent(synthetic):
    zip_path, _ = synthetic
    fetcher = FileFetcher(zip_path)
    entries, cd_off = parse_central_directory_with_offset(fetcher)
    # The central directory begins after every local record.
    max_local = max(e.local_header_offset for e in entries)
    assert cd_off > max_local


# --- Member selection math ----------------------------------------------------


def test_select_subset_50_yields_35_15_per_id(synthetic):
    zip_path, _ = synthetic
    entries = parse_central_directory(FileFetcher(zip_path))
    selected = select_subset(entries, 50)
    # 4 ids * (35 normal + 15 abnormal) = 200.
    assert len(selected) == 200
    per_id = _counts_by_id_condition(selected)
    for id_dir in ("id_00", "id_02", "id_04", "id_06"):
        assert per_id[(id_dir, "normal")] == 35
        assert per_id[(id_dir, "abnormal")] == 15


def test_select_subset_4_yields_3_1_per_id(synthetic):
    zip_path, _ = synthetic
    entries = parse_central_directory(FileFetcher(zip_path))
    selected = select_subset(entries, 4)
    per_id = _counts_by_id_condition(selected)
    for id_dir in ("id_00", "id_02", "id_04", "id_06"):
        assert per_id[(id_dir, "normal")] == 3
        assert per_id[(id_dir, "abnormal")] == 1


def test_select_subset_takes_first_by_sorted_name(synthetic):
    zip_path, _ = synthetic
    entries = parse_central_directory(FileFetcher(zip_path))
    selected = select_subset(entries, 4)
    normal_00 = sorted(
        e.name
        for e in selected
        if e.name.startswith("fan/id_00/normal/")
    )
    assert normal_00 == [
        "fan/id_00/normal/00000000.wav",
        "fan/id_00/normal/00000001.wav",
        "fan/id_00/normal/00000002.wav",
    ]


def _counts_by_id_condition(entries):
    counts: dict[tuple[str, str], int] = {}
    for entry in entries:
        parts = entry.name.split("/")
        key = (parts[1], parts[2])
        counts[key] = counts.get(key, 0) + 1
    return counts


# --- Contiguous-run grouping --------------------------------------------------


def test_group_runs_merges_adjacent_and_splits_on_large_gap():
    e1 = ZipEntry("a", 0, 10, 10, 0)
    e2 = ZipEntry("b", 0, 10, 10, 11)
    e3 = ZipEntry("c", 0, 10, 10, 10_000_000)
    spans = [(0, 10, e1), (11, 20, e2), (10_000_000, 10_000_010, e3)]
    runs = group_runs(spans, gap=1 << 20)
    assert len(runs) == 2
    assert runs[0].entries == (e1, e2)
    assert runs[1].entries == (e3,)


def test_group_runs_merges_within_gap_threshold():
    e1 = ZipEntry("a", 0, 10, 10, 0)
    e2 = ZipEntry("b", 0, 10, 10, 100)  # 89-byte gap < 1 MB
    spans = [(0, 10, e1), (100, 110, e2)]
    runs = group_runs(spans, gap=1 << 20)
    assert len(runs) == 1
    assert runs[0].start == 0 and runs[0].end == 110


def test_compute_member_spans_no_overlap_with_next(synthetic):
    zip_path, _ = synthetic
    fetcher = FileFetcher(zip_path)
    entries, cd_off = parse_central_directory_with_offset(fetcher)
    selected = select_subset(entries, 4)
    spans = compute_member_spans(selected, entries, cd_off)
    for start, end, _entry in spans:
        assert start <= end < cd_off


# --- Extraction correctness ---------------------------------------------------


def test_extract_member_roundtrip_all(synthetic):
    zip_path, contents = synthetic
    fetcher = FileFetcher(zip_path)
    for entry in parse_central_directory(fetcher):
        if not entry.name.endswith(".wav"):
            continue
        assert extract_member(fetcher, entry) == contents[entry.name]


def test_extract_run_member_matches_payload(synthetic):
    zip_path, contents = synthetic
    fetcher = FileFetcher(zip_path)
    entries, cd_off = parse_central_directory_with_offset(fetcher)
    selected = select_subset(entries, 4)
    spans = compute_member_spans(selected, entries, cd_off)
    runs = group_runs(spans)
    extracted: dict[str, bytes] = {}
    for run in runs:
        run_bytes = fetcher.fetch(run.start, run.end)
        for entry in run.entries:
            extracted[entry.name] = extract_run_member(run_bytes, run.start, entry)
    assert len(extracted) == len(selected)
    for name, data in extracted.items():
        assert data == contents[name]


def test_extraction_covers_both_stored_and_deflated(synthetic):
    """Ensure the fixture actually mixes methods, so the test is meaningful."""
    zip_path, _ = synthetic
    methods = {e.method for e in parse_central_directory(FileFetcher(zip_path)) if e.name.endswith(".wav")}
    assert 0 in methods and 8 in methods


# --- Idempotent skip ----------------------------------------------------------


def test_idempotent_skip_via_target_size(synthetic, tmp_path):
    from engine.download import target_is_complete

    zip_path, contents = synthetic
    fetcher = FileFetcher(zip_path)
    entry = next(e for e in parse_central_directory(fetcher) if e.name.endswith(".wav"))
    dest = tmp_path / "out"
    target = member_target_path(dest, "0_dB", entry.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x" * entry.uncomp_size)
    assert target_is_complete(target, entry.uncomp_size) is True
    # Wrong size -> not complete -> would re-fetch.
    target.write_bytes(b"x" * (entry.uncomp_size + 1))
    assert target_is_complete(target, entry.uncomp_size) is False


def test_member_target_path_inserts_snr_level(tmp_path):
    dest = tmp_path / "mimii"
    p = member_target_path(dest, "0_dB", "fan/id_00/normal/00000000.wav")
    assert p == dest / "0_dB" / "fan" / "id_00" / "normal" / "00000000.wav"


# --- Full-mode resume math ----------------------------------------------------


def test_resume_range_header_open_ended():
    assert resume_range_header(0) == {"Range": "bytes=0-"}
    assert resume_range_header(20_000_000) == {"Range": "bytes=20000000-"}


def test_full_mode_resume_requests_correct_range(monkeypatch, tmp_path):
    """A pre-existing .part of size k makes the resume GET send Range: bytes=k-."""
    from engine import download as dl

    part_dir = tmp_path / "zips"
    part_dir.mkdir()
    part_path = part_dir / "0_dB_pump.zip.part"
    existing = b"\x00" * 12345
    part_path.write_bytes(existing)

    captured: dict[str, object] = {}

    remaining = b"REST_OF_FILE_BYTES"
    total = len(existing) + len(remaining)

    class FakeResponse:
        status_code = 206

        def __init__(self):
            self.headers = {
                "content-range": f"bytes {len(existing)}-{total - 1}/{total}",
                "content-length": str(len(remaining)),
            }

        def iter_content(self, chunk_size):
            yield remaining

        def close(self):
            pass

        @property
        def text(self):
            return ""

    class FakeSession:
        headers: dict[str, str] = {}

        def get(self, url, headers, stream, timeout):
            captured["url"] = url
            captured["headers"] = headers
            return FakeResponse()

    requests_made, bytes_fetched = dl._stream_zip_with_resume(
        FakeSession(), "http://example/0_dB_pump.zip/content", part_path
    )

    assert captured["headers"] == {"Range": f"bytes={len(existing)}-"}
    assert bytes_fetched == len(remaining)
    assert part_path.read_bytes() == existing + remaining


# --- HttpFetcher sizing (offline) ---------------------------------------------


def test_http_fetcher_parses_content_range_total(monkeypatch):
    from engine import download as dl

    class FakeResponse:
        status_code = 206
        headers = {"content-range": "bytes 0-0/7869431302"}
        content = b"\x00"

        def close(self):
            pass

        @property
        def text(self):
            return ""

    class FakeSession:
        headers: dict[str, str] = {}

        def get(self, url, headers, stream, timeout):
            return FakeResponse()

    fetcher = HttpFetcher("http://example/file.zip/content", session=FakeSession())
    assert fetcher.size == 7869431302


def test_http_fetcher_retries_then_raises(monkeypatch):
    from engine import download as dl

    # No sleeping in tests.
    monkeypatch.setattr(dl.time, "sleep", lambda *_a, **_k: None)

    class FakeResponse:
        status_code = 503
        headers: dict[str, str] = {}

        def close(self):
            pass

        @property
        def text(self):
            return "busy"

    class FakeSession:
        headers: dict[str, str] = {}

        def __init__(self):
            self.calls = 0

        def get(self, url, headers, stream, timeout):
            self.calls += 1
            return FakeResponse()

    session = FakeSession()
    fetcher = HttpFetcher("http://example/file.zip/content", session=session)
    with pytest.raises(RuntimeError) as exc:
        fetcher.fetch(0, 10)
    assert "file.zip" in str(exc.value)
    assert session.calls == dl.MAX_RETRIES
