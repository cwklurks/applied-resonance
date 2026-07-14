"""Download MIMII (Zenodo record 3384388) zips: full mode or ranged-zip subset mode.

Full mode streams whole zip(s) to ``data/zips/{SNR}_dB_{machine}.zip`` with
Range-based resume, then extracts all WAV members. Subset mode parses the remote
ZIP central directory over HTTP range requests and extracts only a deterministic
subset of clips per machine id, never downloading the whole archive.

Run as: ``uv run python -m engine.download <args>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

import requests

from engine.paths import DATA_DIR, MIMII_DIR

logger = logging.getLogger("engine.download")

# --- Remote constants ---------------------------------------------------------

ZENODO_RECORD = "3384388"
URL_TEMPLATE = (
    "https://zenodo.org/api/records/" + ZENODO_RECORD + "/files/{name}.zip/content"
)
USER_AGENT = "earsight-mimii-downloader/1.0 (+https://zenodo.org/record/3384388)"

SNR_CHOICES = ("-6", "0", "6")
MACHINE_CHOICES = ("fan", "pump")
DEFAULT_SUBSET = 50
DEFAULT_NORMAL_FRACTION = 0.7

# Merge byte-runs separated by gaps smaller than this (1 MB).
RUN_GAP_MERGE_BYTES = 1 << 20
# Stream a run to a temp file rather than RAM when it exceeds this (256 MB).
RUN_STREAM_THRESHOLD = 256 << 20
# Chunk size for streaming HTTP bodies to disk.
STREAM_CHUNK = 1 << 20

# Retry policy for transient remote failures.
MAX_RETRIES = 5
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
BACKOFF_BASE = 1.0
BACKOFF_CAP = 30.0


# --- Byte fetcher abstraction -------------------------------------------------


class Fetcher(Protocol):
    """Reads byte ranges from a source whose total size is known."""

    @property
    def size(self) -> int:
        ...

    def fetch(self, start: int, end: int) -> bytes:
        """Return bytes [start, end] inclusive on both ends."""
        ...


class FileFetcher:
    """Range fetcher backed by a local file. Used for tests (no network)."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._size = self._path.stat().st_size

    @property
    def size(self) -> int:
        return self._size

    def fetch(self, start: int, end: int) -> bytes:
        if start < 0 or end < start or end >= self._size:
            raise ValueError(f"invalid range [{start}, {end}] for size {self._size}")
        with self._path.open("rb") as handle:
            handle.seek(start)
            return handle.read(end - start + 1)


class HttpFetcher:
    """Range fetcher backed by an HTTP endpoint that honours Range requests.

    Size is discovered lazily via a 1-byte ranged GET (Zenodo returns 206 with a
    ``content-range`` header), because HEAD ignores Range on this host.
    """

    def __init__(self, url: str, session: requests.Session | None = None) -> None:
        self._url = url
        self._session = session or requests.Session()
        self._session.headers.setdefault("User-Agent", USER_AGENT)
        self._size: int | None = None
        self.request_count = 0
        self.bytes_fetched = 0

    @property
    def url(self) -> str:
        return self._url

    @property
    def size(self) -> int:
        if self._size is None:
            self._size = self._discover_size()
        return self._size

    def _discover_size(self) -> int:
        response = self._ranged_get(0, 0)
        content_range = response.headers.get("content-range")
        if not content_range or "/" not in content_range:
            raise RuntimeError(
                f"missing content-range while sizing {self._url!r}: {content_range!r}"
            )
        total = content_range.rsplit("/", 1)[1].strip()
        if not total.isdigit():
            raise RuntimeError(f"unparseable content-range total {total!r} for {self._url!r}")
        return int(total)

    def fetch(self, start: int, end: int) -> bytes:
        if start < 0 or end < start:
            raise ValueError(f"invalid range [{start}, {end}]")
        response = self._ranged_get(start, end)
        body = response.content
        self.bytes_fetched += len(body)
        return body

    def stream_to(self, start: int, end: int, handle) -> int:
        """Stream range [start, end] into an open binary file handle.

        Returns the number of bytes written.
        """
        response = self._ranged_get(start, end, stream=True)
        written = 0
        for chunk in response.iter_content(chunk_size=STREAM_CHUNK):
            if chunk:
                handle.write(chunk)
                written += len(chunk)
        self.bytes_fetched += written
        return written

    def _ranged_get(self, start: int, end: int, stream: bool = False) -> requests.Response:
        headers = {"Range": f"bytes={start}-{end}"}
        return self._request(headers, stream=stream, expect_status=206)

    def _request(
        self, headers: dict[str, str], stream: bool, expect_status: int
    ) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            self.request_count += 1
            try:
                response = self._session.get(
                    self._url, headers=headers, stream=stream, timeout=(30, 300)
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                self._sleep_backoff(attempt, retry_after=None)
                continue
            if response.status_code == expect_status:
                return response
            if response.status_code in RETRYABLE_STATUS:
                retry_after = response.headers.get("retry-after")
                response.close()
                last_error = RuntimeError(
                    f"HTTP {response.status_code} from {self._url!r}"
                )
                self._sleep_backoff(attempt, retry_after=retry_after)
                continue
            body_hint = response.text[:200]
            response.close()
            raise RuntimeError(
                f"unexpected HTTP {response.status_code} (wanted {expect_status}) "
                f"from {self._url!r}: {body_hint!r}"
            )
        raise RuntimeError(
            f"exhausted {MAX_RETRIES} retries for {self._url!r}"
        ) from last_error

    def _sleep_backoff(self, attempt: int, retry_after: str | None) -> None:
        if attempt >= MAX_RETRIES - 1:
            return
        delay = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** attempt))
        if retry_after and retry_after.isdigit():
            delay = max(delay, float(retry_after))
        logger.warning("retrying after %.1fs (attempt %d/%d)", delay, attempt + 1, MAX_RETRIES)
        time.sleep(delay)


# --- Central directory parsing ------------------------------------------------


@dataclass(frozen=True)
class ZipEntry:
    """One central-directory record for a ZIP member."""

    name: str
    method: int
    comp_size: int
    uncomp_size: int
    local_header_offset: int


def _read_zip64_extra(extra: bytes, want_uncomp: bool, want_comp: bool, want_offset: bool):
    """Pull the real 64-bit values out of the zip64 extra field (id 0x0001).

    Returns (uncomp, comp, offset) where each is None if not present. The fields
    appear in fixed order (uncomp, comp, offset, disk) but only for those that
    were 0xFFFFFFFF in the fixed record.
    """
    pos = 0
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack("<HH", extra[pos : pos + 4])
        body = extra[pos + 4 : pos + 4 + data_size]
        pos += 4 + data_size
        if header_id != 0x0001:
            continue
        values: list[int] = []
        cursor = 0
        for want in (want_uncomp, want_comp, want_offset):
            if want:
                values.append(struct.unpack("<Q", body[cursor : cursor + 8])[0])
                cursor += 8
        out_uncomp = values.pop(0) if want_uncomp else None
        out_comp = values.pop(0) if want_comp else None
        out_offset = values.pop(0) if want_offset else None
        return out_uncomp, out_comp, out_offset
    return None, None, None


def find_central_directory_extent(fetcher: Fetcher) -> tuple[int, int]:
    """Locate the central directory (offset, size) using the EOCD records.

    Prefers the zip64 EOCD (``PK\\x06\\x06``); falls back to the classic
    EOCD (``PK\\x05\\x06``).
    """
    size = fetcher.size
    tail_len = min(65536, size)
    tail = fetcher.fetch(size - tail_len, size - 1)

    idx64 = tail.rfind(b"PK\x06\x06")
    if idx64 != -1:
        (
            _sig,
            _eocd_size,
            _vmade,
            _vneed,
            _disk,
            _cd_disk,
            _n_disk,
            _n_total,
            cd_size,
            cd_off,
        ) = struct.unpack("<IQHHIIQQQQ", tail[idx64 : idx64 + 56])
        return cd_off, cd_size

    idx = tail.rfind(b"PK\x05\x06")
    if idx == -1:
        raise RuntimeError("no End Of Central Directory record found in archive tail")
    (
        _sig,
        _disk,
        _cd_disk,
        _n_disk,
        _n_total,
        cd_size,
        cd_off,
        _comment_len,
    ) = struct.unpack("<IHHHHIIH", tail[idx : idx + 22])
    return cd_off, cd_size


def parse_central_directory(fetcher: Fetcher) -> list[ZipEntry]:
    """Fetch and parse the full central directory into ZipEntry records."""
    entries, _cd_off = parse_central_directory_with_offset(fetcher)
    return entries


def parse_central_directory_with_offset(fetcher: Fetcher) -> tuple[list[ZipEntry], int]:
    """Like ``parse_central_directory`` but also returns the central-dir offset."""
    cd_off, cd_size = find_central_directory_extent(fetcher)
    cd = fetcher.fetch(cd_off, cd_off + cd_size - 1)
    entries: list[ZipEntry] = []
    pos = 0
    while pos < len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[pos + 10 : pos + 12])[0]
        comp_size = struct.unpack("<I", cd[pos + 20 : pos + 24])[0]
        uncomp_size = struct.unpack("<I", cd[pos + 24 : pos + 28])[0]
        name_len, extra_len, comment_len = struct.unpack("<HHH", cd[pos + 28 : pos + 34])
        local_offset = struct.unpack("<I", cd[pos + 42 : pos + 46])[0]
        name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8")
        extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]

        need_uncomp = uncomp_size == 0xFFFFFFFF
        need_comp = comp_size == 0xFFFFFFFF
        need_offset = local_offset == 0xFFFFFFFF
        if need_uncomp or need_comp or need_offset:
            z_uncomp, z_comp, z_offset = _read_zip64_extra(
                extra, need_uncomp, need_comp, need_offset
            )
            if need_uncomp and z_uncomp is not None:
                uncomp_size = z_uncomp
            if need_comp and z_comp is not None:
                comp_size = z_comp
            if need_offset and z_offset is not None:
                local_offset = z_offset

        entries.append(
            ZipEntry(
                name=name,
                method=method,
                comp_size=comp_size,
                uncomp_size=uncomp_size,
                local_header_offset=local_offset,
            )
        )
        pos += 46 + name_len + extra_len + comment_len
    return entries, cd_off


def entry_to_dict(entry: ZipEntry) -> dict:
    return {
        "name": entry.name,
        "method": entry.method,
        "comp_size": entry.comp_size,
        "uncomp_size": entry.uncomp_size,
        "local_header_offset": entry.local_header_offset,
    }


def entry_from_dict(data: dict) -> ZipEntry:
    return ZipEntry(
        name=data["name"],
        method=data["method"],
        comp_size=data["comp_size"],
        uncomp_size=data["uncomp_size"],
        local_header_offset=data["local_header_offset"],
    )


# --- Member extraction --------------------------------------------------------

LOCAL_HEADER_FIXED = 30


def _decompress(method: int, data: bytes, expected_size: int) -> bytes:
    if method == 0:
        out = data
    elif method == 8:
        out = zlib.decompressobj(-15).decompress(data)
    else:
        raise RuntimeError(f"unsupported compression method {method}")
    if len(out) != expected_size:
        raise RuntimeError(
            f"size mismatch after inflate: got {len(out)}, expected {expected_size}"
        )
    return out


def _local_data_span(local_header: bytes, entry: ZipEntry) -> tuple[int, int]:
    """Given the fixed 30 bytes of the local header, return (data_start_rel, comp_size).

    ``data_start_rel`` is relative to the local_header_offset. The local header's
    name/extra lengths can differ from the central directory's, so we read them
    from the local header (offsets 26 and 28).
    """
    if local_header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header signature for {entry.name!r}")
    name_len, extra_len = struct.unpack("<HH", local_header[26:30])
    return LOCAL_HEADER_FIXED + name_len + extra_len, entry.comp_size


def extract_member(fetcher: Fetcher, entry: ZipEntry) -> bytes:
    """Fetch and decompress a single member via two range requests."""
    header = fetcher.fetch(entry.local_header_offset, entry.local_header_offset + LOCAL_HEADER_FIXED - 1)
    data_rel, comp_size = _local_data_span(header, entry)
    data_start = entry.local_header_offset + data_rel
    if comp_size == 0:
        raw = b""
    else:
        raw = fetcher.fetch(data_start, data_start + comp_size - 1)
    return _decompress(entry.method, raw, entry.uncomp_size)


# --- Subset selection ---------------------------------------------------------


def _is_wav(entry: ZipEntry) -> bool:
    return entry.name.lower().endswith(".wav") and not entry.name.endswith("/")


def _parse_member_path(name: str) -> tuple[str, str, str] | None:
    """Parse ``{machine}/id_{XX}/{normal|abnormal}/file.wav`` into keys.

    Returns (machine_id, condition, filename) or None if the layout doesn't match.
    ``machine_id`` is the ``id_XX`` directory; the leading machine segment is
    dropped because callers already scope by machine.
    """
    parts = name.split("/")
    if len(parts) < 4:
        return None
    *_, id_dir, condition, filename = parts
    if not id_dir.startswith("id_"):
        return None
    if condition not in ("normal", "abnormal"):
        return None
    if not filename.lower().endswith(".wav"):
        return None
    return id_dir, condition, filename


def split_counts(n: int, normal_fraction: float = DEFAULT_NORMAL_FRACTION) -> tuple[int, int]:
    """Split N clips into (n_normal, n_abnormal) with ceil(fraction*N) normal."""
    if n < 0:
        raise ValueError("subset N must be non-negative")
    n_normal = math.ceil(normal_fraction * n)
    n_abnormal = n - n_normal
    return n_normal, n_abnormal


def select_subset(entries: list[ZipEntry], n: int) -> list[ZipEntry]:
    """Pick a deterministic subset: per id_XX, first n_normal + n_abnormal by name."""
    n_normal, n_abnormal = split_counts(n)
    # Group wav members by (id_dir, condition), keeping insertion order stable.
    grouped: dict[tuple[str, str], list[ZipEntry]] = {}
    for entry in entries:
        if not _is_wav(entry):
            continue
        parsed = _parse_member_path(entry.name)
        if parsed is None:
            continue
        id_dir, condition, _filename = parsed
        grouped.setdefault((id_dir, condition), []).append(entry)

    selected: list[ZipEntry] = []
    for (id_dir, condition), members in sorted(grouped.items()):
        members_sorted = sorted(members, key=lambda e: e.name)
        take = n_normal if condition == "normal" else n_abnormal
        selected.extend(members_sorted[:take])
    return selected


# --- Contiguous run grouping --------------------------------------------------


@dataclass(frozen=True)
class Run:
    """A contiguous byte span covering one or more members fetched in one GET."""

    start: int
    end: int  # inclusive
    entries: tuple[ZipEntry, ...]


def compute_member_spans(
    selected: list[ZipEntry], all_entries: list[ZipEntry], cd_off: int
) -> list[tuple[int, int, ZipEntry]]:
    """Compute conservative byte spans for selected members from the central dir.

    No per-member HTTP request is made. The end of each member's local record is
    bounded by the next member's local_header_offset (members are stored
    back-to-back); the final member is bounded by the central directory offset.
    The fetched run is large enough to contain header + data; the exact data
    boundary is then parsed from the local header inside the fetched bytes.
    """
    # Sorted unique local offsets across ALL entries, to find each member's upper bound.
    offsets = sorted({e.local_header_offset for e in all_entries})
    next_offset: dict[int, int] = {}
    for i, off in enumerate(offsets):
        next_offset[off] = offsets[i + 1] if i + 1 < len(offsets) else cd_off

    spans: list[tuple[int, int, ZipEntry]] = []
    for entry in selected:
        start = entry.local_header_offset
        end = next_offset.get(start, cd_off) - 1  # inclusive
        spans.append((start, end, entry))
    return spans


def group_runs(spans: list[tuple[int, int, ZipEntry]], gap: int = RUN_GAP_MERGE_BYTES) -> list[Run]:
    """Merge member byte-spans whose gaps are smaller than ``gap`` into runs.

    ``spans`` is a list of (start, end_inclusive, entry). Output runs are sorted
    by start; members within a run preserve archive order.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: s[0])
    runs: list[Run] = []
    cur_start, cur_end, cur_entry = ordered[0]
    cur_entries: list[ZipEntry] = [cur_entry]
    for start, end, entry in ordered[1:]:
        if start <= cur_end + 1 + gap:
            cur_end = max(cur_end, end)
            cur_entries.append(entry)
        else:
            runs.append(Run(cur_start, cur_end, tuple(cur_entries)))
            cur_start, cur_end, cur_entries = start, end, [entry]
    runs.append(Run(cur_start, cur_end, tuple(cur_entries)))
    return runs


def extract_run_member(run_bytes: bytes, run_start: int, entry: ZipEntry) -> bytes:
    """Extract one member's uncompressed bytes from an in-memory run buffer.

    The local header (whose extra-field length can differ from the central
    directory's) is parsed from the fetched bytes to locate the data exactly.
    """
    local_rel = entry.local_header_offset - run_start
    header = run_bytes[local_rel : local_rel + LOCAL_HEADER_FIXED]
    data_rel, _comp = _local_data_span(header, entry)
    data_off = local_rel + data_rel
    raw = run_bytes[data_off : data_off + entry.comp_size]
    return _decompress(entry.method, raw, entry.uncomp_size)


# --- Output paths & atomic write ----------------------------------------------


def member_target_path(dest: Path, snr_dir: str, name: str) -> Path:
    """Map a zip member name to its on-disk target under ``dest/{SNR}_dB/...``.

    Members are ``{machine}/id_XX/{cond}/file.wav``; the ``{SNR}_dB`` level is
    inserted from the zip name because members don't include it.
    """
    member = PurePosixPath(name)
    raw_parts = name.split("/")
    if (
        not name
        or member.is_absolute()
        or "\\" in name
        or "\x00" in name
        or any(part in ("", ".", "..") for part in raw_parts)
    ):
        raise ValueError(f"unsafe archive member path: {name!r}")

    extraction_root = (dest / snr_dir).resolve()
    target = extraction_root.joinpath(*member.parts).resolve()
    try:
        target.relative_to(extraction_root)
    except ValueError as exc:
        raise ValueError(
            f"archive member path resolves outside extraction root: {name!r}"
        ) from exc
    return target


def write_atomic(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, target)


def target_is_complete(target: Path, expected_size: int) -> bool:
    return target.exists() and target.stat().st_size == expected_size


# --- Cache --------------------------------------------------------------------


def ziplist_cache_path(snr: str, machine: str) -> Path:
    return DATA_DIR / "cache" / f"ziplist_{snr}_dB_{machine}.json"


def load_or_fetch_listing(fetcher: Fetcher, cache_path: Path) -> tuple[list[ZipEntry], int]:
    """Return (entries, cd_off) for the central directory, using a JSON cache."""
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text())
            entries = [entry_from_dict(item) for item in raw["entries"]]
            return entries, raw["cd_off"]
        except (json.JSONDecodeError, KeyError, OSError, TypeError) as exc:
            logger.warning("ignoring corrupt cache %s: %s", cache_path, exc)
    entries, cd_off = parse_central_directory_with_offset(fetcher)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cd_off": cd_off, "entries": [entry_to_dict(e) for e in entries]}
    cache_path.write_text(json.dumps(payload))
    return entries, cd_off


# --- Summary ------------------------------------------------------------------


@dataclass
class DownloadStats:
    machine: str
    snr_dir: str
    files_written: dict[tuple[str, str], int]
    files_skipped: int
    bytes_fetched: int
    requests: int


def _format_stats(stats: DownloadStats) -> str:
    lines = [
        f"[{stats.machine} / {stats.snr_dir}] "
        f"written={sum(stats.files_written.values())} skipped={stats.files_skipped} "
        f"bytes_fetched={stats.bytes_fetched} requests={stats.requests}",
    ]
    for (id_dir, condition), count in sorted(stats.files_written.items()):
        lines.append(f"    {id_dir}/{condition}: {count}")
    return "\n".join(lines)


# --- Subset mode --------------------------------------------------------------


def download_subset(
    url: str,
    machine: str,
    snr: str,
    n: int,
    dest: Path,
    session: requests.Session | None = None,
) -> DownloadStats:
    """Download a deterministic subset of clips for one machine over range GETs."""
    snr_dir = f"{snr}_dB"
    fetcher = HttpFetcher(url, session=session)
    cache_path = ziplist_cache_path(snr, machine)
    entries, cd_off = load_or_fetch_listing(fetcher, cache_path)

    selected = select_subset(entries, n)

    # Idempotent skip: drop members already on disk at the right size.
    pending: list[ZipEntry] = []
    skipped = 0
    for entry in selected:
        target = member_target_path(dest, snr_dir, entry.name)
        if target_is_complete(target, entry.uncomp_size):
            skipped += 1
        else:
            pending.append(entry)

    files_written: dict[tuple[str, str], int] = {}

    if pending:
        spans = compute_member_spans(pending, entries, cd_off)
        runs = group_runs(spans)
        for run in runs:
            _process_run(fetcher, run, dest, snr_dir, files_written)

    return DownloadStats(
        machine=machine,
        snr_dir=snr_dir,
        files_written=files_written,
        files_skipped=skipped,
        bytes_fetched=fetcher.bytes_fetched,
        requests=fetcher.request_count,
    )


def _process_run(
    fetcher: HttpFetcher,
    run: Run,
    dest: Path,
    snr_dir: str,
    files_written: dict[tuple[str, str], int],
) -> None:
    """Fetch one run and write each member, streaming to temp file when large."""
    run_size = run.end - run.start + 1
    if run_size > RUN_STREAM_THRESHOLD:
        _process_run_streamed(fetcher, run, dest, snr_dir, files_written)
        return
    run_bytes = fetcher.fetch(run.start, run.end)
    for entry in run.entries:
        data = extract_run_member(run_bytes, run.start, entry)
        _write_member(entry, data, dest, snr_dir, files_written)


def _process_run_streamed(
    fetcher: HttpFetcher,
    run: Run,
    dest: Path,
    snr_dir: str,
    files_written: dict[tuple[str, str], int],
) -> None:
    """Stream a large run to a temp file, then slice members out of it."""
    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f"run_{run.start}_{run.end}.bin.part"
    try:
        with tmp.open("wb") as handle:
            fetcher.stream_to(run.start, run.end, handle)
        with tmp.open("rb") as handle:
            for entry in run.entries:
                local_rel = entry.local_header_offset - run.start
                handle.seek(local_rel)
                header = handle.read(LOCAL_HEADER_FIXED)
                data_rel, _comp = _local_data_span(header, entry)
                handle.seek(local_rel + data_rel)
                raw = handle.read(entry.comp_size)
                data = _decompress(entry.method, raw, entry.uncomp_size)
                _write_member(entry, data, dest, snr_dir, files_written)
    finally:
        tmp.unlink(missing_ok=True)


def _write_member(
    entry: ZipEntry,
    data: bytes,
    dest: Path,
    snr_dir: str,
    files_written: dict[tuple[str, str], int],
) -> None:
    target = member_target_path(dest, snr_dir, entry.name)
    write_atomic(target, data)
    parsed = _parse_member_path(entry.name)
    if parsed is not None:
        id_dir, condition, _filename = parsed
        key = (id_dir, condition)
        files_written[key] = files_written.get(key, 0) + 1


# --- Full mode ----------------------------------------------------------------


def _zip_part_path(snr: str, machine: str) -> Path:
    return DATA_DIR / "zips" / f"{snr}_dB_{machine}.zip"


def resume_range_header(part_size: int) -> dict[str, str]:
    """Build the resume Range header for an existing partial download."""
    return {"Range": f"bytes={part_size}-"}


def download_full(
    url: str,
    machine: str,
    snr: str,
    dest: Path,
    session: requests.Session | None = None,
) -> DownloadStats:
    """Stream the whole zip with Range resume, then extract all WAV members."""
    snr_dir = f"{snr}_dB"
    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", USER_AGENT)

    zip_path = _zip_part_path(snr, machine)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = zip_path.with_name(zip_path.name + ".part")

    request_count = 0
    bytes_fetched = 0

    if not zip_path.exists():
        request_count, bytes_fetched = _stream_zip_with_resume(sess, url, part_path)
        os.replace(part_path, zip_path)

    fetcher = FileFetcher(zip_path)
    entries = parse_central_directory(fetcher)
    files_written: dict[tuple[str, str], int] = {}
    skipped = 0
    for entry in entries:
        if not _is_wav(entry):
            continue
        target = member_target_path(dest, snr_dir, entry.name)
        if target_is_complete(target, entry.uncomp_size):
            skipped += 1
            continue
        data = extract_member(fetcher, entry)
        _write_member(entry, data, dest, snr_dir, files_written)

    return DownloadStats(
        machine=machine,
        snr_dir=snr_dir,
        files_written=files_written,
        files_skipped=skipped,
        bytes_fetched=bytes_fetched,
        requests=request_count,
    )


def _stream_zip_with_resume(
    session: requests.Session, url: str, part_path: Path
) -> tuple[int, int]:
    """Stream a zip to ``part_path``, resuming from an existing partial.

    Returns (requests_made, bytes_fetched_this_run). Verifies the final size
    against the server's reported total.
    """
    part_size = part_path.stat().st_size if part_path.exists() else 0
    request_count = 0
    bytes_fetched = 0
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        part_size = part_path.stat().st_size if part_path.exists() else 0
        headers = resume_range_header(part_size)
        if part_size:
            logger.info("resuming download from byte offset %d", part_size)
        request_count += 1
        try:
            response = session.get(url, headers=headers, stream=True, timeout=(30, 300))
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            _sleep_backoff_simple(attempt)
            continue

        if response.status_code not in (200, 206):
            if response.status_code == 416:
                # Already have all bytes; treat as complete.
                response.close()
                break
            if response.status_code in RETRYABLE_STATUS:
                retry_after = response.headers.get("retry-after")
                response.close()
                last_error = RuntimeError(f"HTTP {response.status_code} from {url!r}")
                _sleep_backoff_simple(attempt, retry_after)
                continue
            body_hint = response.text[:200]
            response.close()
            raise RuntimeError(
                f"unexpected HTTP {response.status_code} from {url!r}: {body_hint!r}"
            )

        total = _content_total(response, part_size)
        mode = "ab" if (part_size and response.status_code == 206) else "wb"
        if mode == "wb":
            part_size = 0
        try:
            with part_path.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=STREAM_CHUNK):
                    if chunk:
                        handle.write(chunk)
                        bytes_fetched += len(chunk)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            _sleep_backoff_simple(attempt)
            continue
        finally:
            response.close()

        final_size = part_path.stat().st_size
        if total is not None and final_size != total:
            last_error = RuntimeError(
                f"incomplete download {final_size}/{total} bytes for {url!r}"
            )
            _sleep_backoff_simple(attempt)
            continue
        return request_count, bytes_fetched

    raise RuntimeError(f"failed to download {url!r} after {MAX_RETRIES} attempts") from last_error


def _content_total(response: requests.Response, part_size: int) -> int | None:
    content_range = response.headers.get("content-range")
    if content_range and "/" in content_range:
        tail = content_range.rsplit("/", 1)[1].strip()
        if tail.isdigit():
            return int(tail)
    length = response.headers.get("content-length")
    if length and length.isdigit():
        if response.status_code == 206:
            return part_size + int(length)
        return int(length)
    return None


def _sleep_backoff_simple(attempt: int, retry_after: str | None = None) -> None:
    if attempt >= MAX_RETRIES - 1:
        return
    delay = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** attempt))
    if retry_after and retry_after.isdigit():
        delay = max(delay, float(retry_after))
    logger.warning("retrying after %.1fs (attempt %d/%d)", delay, attempt + 1, MAX_RETRIES)
    time.sleep(delay)


# --- CLI ----------------------------------------------------------------------


def build_url(snr: str, machine: str) -> str:
    return URL_TEMPLATE.format(name=f"{snr}_dB_{machine}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="engine.download",
        description="Download MIMII zips (full or ranged-zip subset mode).",
    )
    parser.add_argument(
        "--machines",
        nargs="+",
        choices=MACHINE_CHOICES,
        default=list(MACHINE_CHOICES),
        help="machines to download (default: both)",
    )
    parser.add_argument(
        "--snr",
        choices=SNR_CHOICES,
        default="0",
        help="signal-to-noise ratio in dB (default: 0)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="subset mode: N clips per machine id (default full mode; "
        f"omitted-but-subset default is {DEFAULT_SUBSET})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=MIMII_DIR,
        help=f"output root (default: {MIMII_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    all_stats: list[DownloadStats] = []
    for machine in args.machines:
        url = build_url(args.snr, machine)
        if args.subset is not None:
            logger.info("subset mode: %s clips per id for %s", args.subset, machine)
            stats = download_subset(url, machine, args.snr, args.subset, dest, session)
        else:
            logger.info("full mode: %s", machine)
            stats = download_full(url, machine, args.snr, dest, session)
        all_stats.append(stats)
        print(_format_stats(stats))

    total_bytes = sum(s.bytes_fetched for s in all_stats)
    total_requests = sum(s.requests for s in all_stats)
    total_written = sum(sum(s.files_written.values()) for s in all_stats)
    total_skipped = sum(s.files_skipped for s in all_stats)
    print(
        f"TOTAL: written={total_written} skipped={total_skipped} "
        f"bytes_fetched={total_bytes} requests={total_requests}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
