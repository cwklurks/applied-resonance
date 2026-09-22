"""Helpers for killtest machine-id selection."""

from __future__ import annotations

from typing import Sequence

IdKey = tuple[str, str]


def format_id_key(key: IdKey) -> str:
    """Render ``("fan", "id_06")`` as ``"fan:id_06"``."""
    return f"{key[0]}:{key[1]}"


def parse_id_tokens(
    tokens: Sequence[str] | None,
    *,
    option_name: str = "--ids",
) -> tuple[IdKey, ...]:
    """Parse CLI id tokens like ``fan:id_06`` into stable id keys."""
    if not tokens:
        return ()

    parsed: list[IdKey] = []
    seen: set[IdKey] = set()
    for token in tokens:
        if token.count(":") != 1:
            raise ValueError(
                f"{option_name} values must look like machine:id, got {token!r}"
            )
        machine, machine_id = token.split(":", 1)
        if not machine or not machine_id:
            raise ValueError(
                f"{option_name} values must look like machine:id, got {token!r}"
            )
        key = (machine, machine_id)
        if key in seen:
            raise ValueError(f"duplicate {option_name} value: {token!r}")
        seen.add(key)
        parsed.append(key)

    return tuple(parsed)
