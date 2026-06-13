"""Rabbit-hole flag derivation (MCPC-032).

Flags are pure, server-side derivations attached to each roster entry as badges
only — NEVER acted on. They surface "is this agent making progress or stuck?"
signals for the dashboard:

- ``looping`` — the same tool name + args hash repeated >= a threshold within a
  window (default 5x in 10 min). The repeat count is tracked per session in the
  middleware's recent-tool-call ring; this module only thresholds it.
- ``play-idle`` — the session holds the play lease but no owner activity for a
  while (default 2 min of lease inactivity).
- ``long-op`` — a single in-flight call has been running or parked beyond a
  threshold (default 10 min).
- ``parked-over-budget`` — a call is currently parked past its wait budget
  (i.e. the park has lasted longer than the gate's park budget would allow a
  single park attempt — a sign of repeated re-park churn).
- ``intent-stale`` — the intent line has not changed while activity continues
  for a long time (default 15 min).

All thresholds are env-overridable with the documented defaults. The derivation
reads pre-computed inputs (a small dataclass the roster builder assembles) so it
has no I/O and cannot fail a roster push.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FLAG_LOOPING = "looping"
FLAG_PLAY_IDLE = "play-idle"
FLAG_LONG_OP = "long-op"
FLAG_PARKED_OVER_BUDGET = "parked-over-budget"
FLAG_INTENT_STALE = "intent-stale"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %.1f", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %d", name, raw, default)
        return default


def loop_repeat_threshold() -> int:
    return _env_int("UNITY_MCP_FLAG_LOOP_REPEATS", 5)


def play_idle_threshold_s() -> float:
    return _env_float("UNITY_MCP_FLAG_PLAY_IDLE_S", 120.0)


def long_op_threshold_s() -> float:
    return _env_float("UNITY_MCP_FLAG_LONG_OP_S", 600.0)


def parked_over_budget_threshold_s() -> float:
    # The gate's single-park budget defaults to 15s; a park observed beyond this
    # means the call has re-parked repeatedly (churn), not a single bounded wait.
    return _env_float("UNITY_MCP_FLAG_PARKED_OVER_BUDGET_S", 15.0)


def intent_stale_threshold_s() -> float:
    return _env_float("UNITY_MCP_FLAG_INTENT_STALE_S", 900.0)


@dataclass
class FlagInputs:
    """Pre-computed per-session inputs for flag derivation.

    All durations are in seconds; ``None`` means "not applicable / unknown" and
    suppresses the corresponding flag (fail open: absent data never flags).
    """

    loop_repeat_count: int = 0
    holds_play_lease: bool = False
    play_lease_idle_seconds: float | None = None
    in_flight_call_age_seconds: float | None = None
    parked_age_seconds: float | None = None
    intent_unchanged_seconds: float | None = None
    activity_recent: bool = False


def derive_flags(inputs: FlagInputs) -> list[str]:
    """Return the sorted list of rabbit-hole flag badges for one session."""
    flags: list[str] = []
    try:
        if inputs.loop_repeat_count >= loop_repeat_threshold():
            flags.append(FLAG_LOOPING)

        if (
            inputs.holds_play_lease
            and inputs.play_lease_idle_seconds is not None
            and inputs.play_lease_idle_seconds >= play_idle_threshold_s()
        ):
            flags.append(FLAG_PLAY_IDLE)

        if (
            inputs.in_flight_call_age_seconds is not None
            and inputs.in_flight_call_age_seconds >= long_op_threshold_s()
        ):
            flags.append(FLAG_LONG_OP)

        if (
            inputs.parked_age_seconds is not None
            and inputs.parked_age_seconds >= parked_over_budget_threshold_s()
        ):
            flags.append(FLAG_PARKED_OVER_BUDGET)

        # Intent staleness only matters while the agent is still doing things —
        # an idle agent's unchanging intent is just idleness, not a rabbit hole.
        if (
            inputs.activity_recent
            and inputs.intent_unchanged_seconds is not None
            and inputs.intent_unchanged_seconds >= intent_stale_threshold_s()
        ):
            flags.append(FLAG_INTENT_STALE)
    except Exception as exc:  # pure derivation must never break a roster push
        logger.debug("rabbit_hole_flags: derivation failed open: %r", exc)
        return []
    return sorted(flags)
