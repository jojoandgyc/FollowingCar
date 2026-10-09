"""Capture-clock diagnostics, separate from processing budgets and authority.

All inputs are monotonic timestamps from the same clock. These values only
describe elapsed time; they never determine frame admission or extend a lease.
"""
import math


def capture_pipeline_timing(*, capture_timestamp, pipeline_started_at,
                            vision_finished_at, decision_finished_at):
    """Report capture ages without presenting an invalid clock as zero age.

    The legacy vision/result age fields describe processing duration and can
    remain unchanged. Callers append these distinct fields to their diagnostic
    record. A future capture or reversed endpoint invalidates the entire clock
    chain, rather than mixing apparently valid ages from different clocks.
    """
    values = (capture_timestamp, pipeline_started_at,
              vision_finished_at, decision_finished_at)
    valid = bool(
        all(isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) for value in values)
        and 0 < capture_timestamp <= pipeline_started_at
        <= vision_finished_at <= decision_finished_at
    )
    return {
        "capture_clock_valid": valid,
        "capture_to_pipeline_ms": (
            (pipeline_started_at - capture_timestamp) * 1000.0 if valid else None),
        "capture_result_age_ms": (
            (vision_finished_at - capture_timestamp) * 1000.0 if valid else None),
        "capture_decision_age_ms": (
            (decision_finished_at - capture_timestamp) * 1000.0 if valid else None),
    }
