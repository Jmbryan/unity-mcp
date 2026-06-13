"""Integration tests for rabbit-hole flag derivation (MCPC-032).

Each flag fires above its (env-overridable) threshold and stays silent below it.
Derivation is pure: absent/None inputs never flag (fail open).
"""

import pytest

from services.state.rabbit_hole_flags import (
    FLAG_INTENT_STALE,
    FLAG_LONG_OP,
    FLAG_LOOPING,
    FLAG_PARKED_OVER_BUDGET,
    FLAG_PLAY_IDLE,
    FlagInputs,
    derive_flags,
)


class TestLooping:
    def test_fires_at_threshold(self):
        assert FLAG_LOOPING in derive_flags(FlagInputs(loop_repeat_count=5))

    def test_silent_below_threshold(self):
        assert FLAG_LOOPING not in derive_flags(FlagInputs(loop_repeat_count=4))

    def test_threshold_env_override(self, monkeypatch):
        monkeypatch.setenv("UNITY_MCP_FLAG_LOOP_REPEATS", "3")
        assert FLAG_LOOPING in derive_flags(FlagInputs(loop_repeat_count=3))


class TestPlayIdle:
    def test_fires_when_lease_held_and_idle(self):
        flags = derive_flags(
            FlagInputs(holds_play_lease=True, play_lease_idle_seconds=130.0)
        )
        assert FLAG_PLAY_IDLE in flags

    def test_silent_when_not_idle(self):
        flags = derive_flags(
            FlagInputs(holds_play_lease=True, play_lease_idle_seconds=10.0)
        )
        assert FLAG_PLAY_IDLE not in flags

    def test_silent_without_lease(self):
        flags = derive_flags(
            FlagInputs(holds_play_lease=False, play_lease_idle_seconds=9999.0)
        )
        assert FLAG_PLAY_IDLE not in flags


class TestLongOp:
    def test_fires_above_threshold(self):
        assert FLAG_LONG_OP in derive_flags(
            FlagInputs(in_flight_call_age_seconds=601.0)
        )

    def test_silent_below_threshold(self):
        assert FLAG_LONG_OP not in derive_flags(
            FlagInputs(in_flight_call_age_seconds=60.0)
        )

    def test_silent_when_no_call(self):
        assert FLAG_LONG_OP not in derive_flags(FlagInputs())


class TestParkedOverBudget:
    def test_fires_above_budget(self):
        assert FLAG_PARKED_OVER_BUDGET in derive_flags(
            FlagInputs(parked_age_seconds=16.0)
        )

    def test_silent_below_budget(self):
        assert FLAG_PARKED_OVER_BUDGET not in derive_flags(
            FlagInputs(parked_age_seconds=5.0)
        )


class TestIntentStale:
    def test_fires_when_stale_and_active(self):
        flags = derive_flags(
            FlagInputs(intent_unchanged_seconds=901.0, activity_recent=True)
        )
        assert FLAG_INTENT_STALE in flags

    def test_silent_when_idle(self):
        # Unchanging intent while idle is just idleness, not a rabbit hole.
        flags = derive_flags(
            FlagInputs(intent_unchanged_seconds=9999.0, activity_recent=False)
        )
        assert FLAG_INTENT_STALE not in flags

    def test_silent_below_threshold(self):
        flags = derive_flags(
            FlagInputs(intent_unchanged_seconds=100.0, activity_recent=True)
        )
        assert FLAG_INTENT_STALE not in flags


def test_no_flags_for_empty_inputs():
    assert derive_flags(FlagInputs()) == []


def test_multiple_flags_sorted():
    flags = derive_flags(
        FlagInputs(
            loop_repeat_count=10,
            in_flight_call_age_seconds=999.0,
        )
    )
    assert flags == sorted(flags)
    assert FLAG_LOOPING in flags and FLAG_LONG_OP in flags
