"""Tests for the explicit LISTENING -> SUSPECT -> ALERT alert policy.

Each scenario drives ``AlertPolicy.update(percentile, t)`` once per synthetic
1 s window and asserts the resulting state sequence. The policy is the
auditable safety core, so the tests pin EXACT transition timing.
"""

import pytest

from engine.policy import AlertPolicy, PolicyConfig, State


def run(policy: AlertPolicy, percentiles, t0: float = 0.0, dt: float = 1.0):
    """Feed percentiles at evenly spaced times, return the list of states."""
    return [policy.update(p, t0 + i * dt) for i, p in enumerate(percentiles)]


# --- Scenario 1: brief spike never alerts ----------------------------------


def test_brief_spike_reaches_suspect_but_never_alerts():
    policy = AlertPolicy()
    states = run(policy, [50, 99.5, 99.5, 50, 50, 50])

    assert states == [
        State.LISTENING,
        State.SUSPECT,
        State.SUSPECT,
        State.LISTENING,
        State.LISTENING,
        State.LISTENING,
    ]
    assert State.ALERT not in states


# --- Scenario 2: sustained anomaly alerts at exactly window n ----------------


def test_sustained_anomaly_alerts_at_exactly_n_consecutive():
    policy = AlertPolicy()
    states = run(policy, [99.5] * 8)

    # SUSPECT for windows 1..4 (indices 0..3), ALERT from window 5 (index 4).
    assert states[:4] == [State.SUSPECT] * 4
    assert states[4] == State.ALERT
    assert all(s == State.ALERT for s in states[4:])
    # First ALERT lands at exactly the n_consecutive-th window.
    assert states.index(State.ALERT) == PolicyConfig().n_consecutive - 1


# --- Scenario 3: flapping around the enter threshold never alerts -----------


def test_flapping_around_enter_threshold_never_alerts():
    policy = AlertPolicy()
    sequence = [99.5, 98.0] * 20
    states = run(policy, sequence)

    assert State.ALERT not in states
    # Hot windows -> SUSPECT, dips below enter -> back to LISTENING.
    expected = [State.SUSPECT, State.LISTENING] * 20
    assert states == expected
    assert states.count(State.ALERT) == 0


# --- Scenario 4: hysteresis holds ALERT in the [exit, enter) band -----------


def test_hysteresis_holds_alert_until_below_exit():
    policy = AlertPolicy()
    # Drive to ALERT.
    run(policy, [99.5] * 5)
    assert policy.state == State.ALERT

    # pct=97 is in [exit=95, enter=99): stays ALERT for 10 windows.
    hold = run(policy, [97.0] * 10, t0=5.0)
    assert hold == [State.ALERT] * 10

    # Drop below exit -> LISTENING.
    after = policy.update(94.0, 15.0)
    assert after == State.LISTENING


# --- Scenario 5: cooldown suppresses re-ALERT until it elapses ---------------


def test_cooldown_suppresses_realert_until_elapsed():
    policy = AlertPolicy()
    # Drive to ALERT over t=0..4.
    run(policy, [99.5] * 5)
    assert policy.state == State.ALERT

    # Exit ALERT at t=5 -> cooldown_until = 5 + 30 = 35.
    assert policy.update(94.0, 5.0) == State.LISTENING

    # Sustain 99.5 from t=6: SUSPECT but suppressed while t < 35,
    # even once the consecutive streak is >= n_consecutive.
    suppressed = run(policy, [99.5] * 20, t0=6.0)  # t = 6..25
    assert State.ALERT not in suppressed
    assert all(s == State.SUSPECT for s in suppressed)

    # Still suppressed strictly before cooldown_until.
    assert policy.update(99.5, 34.0) == State.SUSPECT
    # First update at t >= cooldown_until with streak intact -> ALERT.
    assert policy.update(99.5, 35.0) == State.ALERT


# --- Scenario 6: custom config ----------------------------------------------


def test_custom_config_alerts_at_window_two_and_honors_cooldown():
    config = PolicyConfig(
        n_consecutive=2,
        enter_percentile=90.0,
        exit_percentile=80.0,
        cooldown_s=5.0,
    )
    policy = AlertPolicy(config)

    # Sustained anomaly alerts at window 2 (index 1).
    states = run(policy, [95.0] * 4)
    assert states[0] == State.SUSPECT
    assert states[1] == State.ALERT
    assert states.index(State.ALERT) == config.n_consecutive - 1

    # Exit at t=4 (below exit=80) -> cooldown_until = 4 + 5 = 9.
    assert policy.update(70.0, 4.0) == State.LISTENING

    # Re-arm: SUSPECT but suppressed before t=9.
    assert policy.update(95.0, 5.0) == State.SUSPECT
    assert policy.update(95.0, 8.0) == State.SUSPECT
    # First update at the 5 s mark (t >= 9) with streak intact -> ALERT.
    assert policy.update(95.0, 9.0) == State.ALERT


# --- Scenario 7: validation --------------------------------------------------


def test_percentile_above_100_raises():
    policy = AlertPolicy()
    with pytest.raises(ValueError):
        policy.update(101.0, 0.0)


def test_percentile_below_0_raises():
    policy = AlertPolicy()
    with pytest.raises(ValueError):
        policy.update(-1.0, 0.0)


def test_decreasing_time_raises_and_names_both_values():
    policy = AlertPolicy()
    policy.update(50.0, 10.0)
    with pytest.raises(ValueError) as exc:
        policy.update(50.0, 9.0)
    message = str(exc.value)
    assert "9" in message and "10" in message


def test_config_with_exit_above_enter_raises():
    with pytest.raises(ValueError):
        PolicyConfig(enter_percentile=95.0, exit_percentile=99.0)


def test_config_invalid_n_consecutive_raises():
    with pytest.raises(ValueError):
        PolicyConfig(n_consecutive=0)


def test_config_negative_cooldown_raises():
    with pytest.raises(ValueError):
        PolicyConfig(cooldown_s=-1.0)


def test_config_enter_above_100_raises():
    with pytest.raises(ValueError):
        PolicyConfig(enter_percentile=101.0)


def test_config_negative_exit_raises():
    with pytest.raises(ValueError):
        PolicyConfig(exit_percentile=-1.0)


def test_reset_returns_to_listening_with_cleared_counters_and_cooldown():
    policy = AlertPolicy()
    # Drive to ALERT then exit to set a cooldown.
    run(policy, [99.5] * 5)
    policy.update(94.0, 5.0)
    assert policy.state == State.LISTENING  # exited, but cooldown is armed

    policy.reset()
    assert policy.state == State.LISTENING

    # After reset there is no lingering cooldown: a fresh sustained anomaly
    # alerts at exactly n_consecutive (index 4), proving counters cleared.
    states = run(policy, [99.5] * 6, t0=0.0)
    assert states.index(State.ALERT) == PolicyConfig().n_consecutive - 1

    # And time tracking is reset: starting again at t=0 must not raise.
    policy.reset()
    assert policy.update(50.0, 0.0) == State.LISTENING


# --- Scenario 8: determinism -------------------------------------------------


def test_determinism_same_sequence_yields_identical_states():
    sequence = [50, 99.5, 99.5, 99.5, 99.5, 99.5, 97, 97, 94, 99.5, 99.5]

    first = run(AlertPolicy(), sequence)
    second = run(AlertPolicy(), sequence)

    assert first == second
