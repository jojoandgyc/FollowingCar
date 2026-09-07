#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.lateral_intent import (
    LateralControlIntent,
    LateralIntentStore,
    slew_signed_rpm,
)


def make_intent(*, published_at: float = 10.0) -> LateralControlIntent:
    return LateralControlIntent(
        sequence=0,
        target_id=7,
        frame_index=100,
        published_at=published_at,
        valid_until=published_at + 0.15,
        x_ratio=0.60,
        motion_dx_ratio=0.02,
        target_image_rate_dps=12.0,
        mode="forward",
        base_percent=40,
        base_rpm=40,
        initial_correction_rpm=5,
        correction_limit_rpm=14.0,
        confidence=0.9,
        bbox_quality="reliable",
        reason="test",
    )


def main() -> int:
    store = LateralIntentStore()
    first = store.publish(make_intent())
    second = store.publish(make_intent(published_at=10.02))
    if first.sequence != 1 or second.sequence != 2:
        raise AssertionError((first.sequence, second.sequence))
    if store.snapshot() != second:
        raise AssertionError("latest-value store did not publish atomically")
    if not second.valid(10.16) or second.valid(10.18):
        raise AssertionError("intent TTL boundary is wrong")

    projected, horizon = second.projected_x_ratio(
        10.12,
        camera_hfov_deg=60.0,
        max_projection_sec=0.12,
        max_projection_ratio=0.10,
    )
    if abs(horizon - 0.10) > 1e-9 or abs(projected - 0.62) > 1e-9:
        raise AssertionError((projected, horizon))
    capped, capped_horizon = second.projected_x_ratio(
        11.0,
        camera_hfov_deg=60.0,
        max_projection_sec=0.12,
        max_projection_ratio=0.01,
    )
    if abs(capped_horizon - 0.12) > 1e-9 or abs(capped - 0.61) > 1e-9:
        raise AssertionError((capped, capped_horizon))

    rising = slew_signed_rpm(
        0,
        14,
        1.0 / 30.0,
        rise_rpm_per_sec=120.0,
        brake_rpm_per_sec=220.0,
    )
    braking = slew_signed_rpm(
        14,
        0,
        1.0 / 30.0,
        rise_rpm_per_sec=120.0,
        brake_rpm_per_sec=220.0,
    )
    reversing = slew_signed_rpm(
        6,
        -6,
        1.0 / 30.0,
        rise_rpm_per_sec=120.0,
        brake_rpm_per_sec=220.0,
    )
    if rising != 4 or braking != 7 or reversing != 0:
        raise AssertionError((rising, braking, reversing))

    first_fine_left = slew_signed_rpm(
        0,
        -1,
        1.0 / 30.0,
        rise_rpm_per_sec=12.0,
        brake_rpm_per_sec=220.0,
    )
    first_fine_right = slew_signed_rpm(
        0,
        1,
        1.0 / 30.0,
        rise_rpm_per_sec=12.0,
        brake_rpm_per_sec=220.0,
    )
    if first_fine_left != -1 or first_fine_right != 1:
        raise AssertionError((first_fine_left, first_fine_right))

    if store.clear() != second or store.snapshot() is not None:
        raise AssertionError("intent clear failed")
    print("lateral_intent_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
