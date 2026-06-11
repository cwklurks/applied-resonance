"""Explicit LISTENING -> SUSPECT -> ALERT alert state machine.

This is the auditable safety core for the technician HUD. A false ALERT costs
trust, so the policy is deliberately conservative: a sustained anomaly must
span ``n_consecutive`` analysis windows before ALERT, hysteresis holds the
ALERT through a noisy band on the way down, and a cooldown suppresses a fresh
ALERT for a fixed period after one clears.

Dependency-free (stdlib only). No wall-clock reads: time is supplied by the
caller via ``t`` (seconds, monotonically non-decreasing) on every ``update``.
"""

import enum
from dataclasses import dataclass


class State(enum.Enum):
    LISTENING = "LISTENING"
    SUSPECT = "SUSPECT"
    ALERT = "ALERT"


@dataclass(frozen=True)
class PolicyConfig:
    """Tunable thresholds for the alert state machine.

    Attributes:
        n_consecutive: Windows at or above ``enter_percentile`` required to
            transition into ALERT (must be >= 1).
        enter_percentile: A window at or above this is "hot" and arms/advances
            the SUSPECT streak.
        exit_percentile: ALERT is held (hysteresis) until a window drops below
            this. Must satisfy ``0 <= exit < enter <= 100``.
        cooldown_s: After an ALERT clears, no new ALERT until this many seconds
            have elapsed (must be >= 0).
    """

    n_consecutive: int = 5
    enter_percentile: float = 99.0
    exit_percentile: float = 95.0
    cooldown_s: float = 30.0

    def __post_init__(self) -> None:
        if self.n_consecutive < 1:
            raise ValueError(
                f"n_consecutive must be >= 1, got {self.n_consecutive}"
            )
        if not (
            0 <= self.exit_percentile < self.enter_percentile <= 100
        ):
            raise ValueError(
                "thresholds must satisfy 0 <= exit_percentile < "
                f"enter_percentile <= 100, got exit={self.exit_percentile}, "
                f"enter={self.enter_percentile}"
            )
        if self.cooldown_s < 0:
            raise ValueError(
                f"cooldown_s must be >= 0, got {self.cooldown_s}"
            )


class AlertPolicy:
    """Conservative three-state alert policy driven one window at a time."""

    def __init__(self, config: PolicyConfig = PolicyConfig()) -> None:
        self._config = config
        self.reset()

    @property
    def state(self) -> State:
        return self._state

    def reset(self) -> None:
        """Clear all runtime state: back to LISTENING, no streak, no cooldown,
        no recorded time."""
        self._state = State.LISTENING
        self._consecutive = 0
        self._cooldown_until = 0.0
        self._last_t: float | None = None

    def update(self, percentile: float, t: float) -> State:
        """Advance the state machine by one analysis window.

        Args:
            percentile: This window's score in [0, 100] vs the machine's own
                baseline.
            t: Window time in seconds, monotonically non-decreasing across
                calls.

        Returns:
            The state after processing this window.

        Raises:
            ValueError: If ``percentile`` is outside [0, 100], or if ``t``
                decreases relative to the previous call.
        """
        if not (0 <= percentile <= 100):
            raise ValueError(
                f"percentile must be in [0, 100], got {percentile}"
            )
        if self._last_t is not None and t < self._last_t:
            raise ValueError(
                f"t must not decrease: got {t} after {self._last_t}"
            )
        self._last_t = t

        config = self._config
        hot = percentile >= config.enter_percentile

        if self._state is State.LISTENING:
            if hot:
                # A single hot window is already worth surfacing as SUSPECT.
                self._state = State.SUSPECT
                self._consecutive = 1
            else:
                self._consecutive = 0

        elif self._state is State.SUSPECT:
            if hot:
                self._consecutive += 1
                ready = self._consecutive >= config.n_consecutive
                cooled_down = t >= self._cooldown_until
                if ready and cooled_down:
                    self._state = State.ALERT
                # If still in cooldown, remain SUSPECT; the count keeps
                # accruing so the first non-suppressed hot window alerts.
            else:
                # No hysteresis on SUSPECT; any dip resets the streak.
                self._state = State.LISTENING
                self._consecutive = 0

        elif self._state is State.ALERT:
            if percentile >= config.exit_percentile:
                # Hysteresis band [exit, enter) holds the alert.
                pass
            else:
                self._state = State.LISTENING
                self._consecutive = 0
                self._cooldown_until = t + config.cooldown_s

        return self._state
