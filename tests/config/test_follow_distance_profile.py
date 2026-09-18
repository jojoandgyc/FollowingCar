"""Only import configuration: never construct PersonTracker or start hardware."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def profile(**overrides):
    env = dict(os.environ, **overrides)
    result = subprocess.run(
        [sys.executable, "-c", "import json; import request_0513_modular as r; "
         "print(json.dumps([r.TARGET_DISTANCE,r.FOLLOW_FORWARD_START_DISTANCE_M,"
         "r.FOLLOW_FORWARD_STOP_DISTANCE_M,r.FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_DISTANCE_M,"
         "r.DISTANCE_PID_DEADBAND_M]))"],
        cwd=ROOT, env=env, check=True, text=True, capture_output=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("distance", [1.5, 1.6, 2.0])
def test_one_distance_setting_derives_small_hysteresis(distance):
    values = profile(TARGET_DISTANCE=str(distance), FOLLOW_DISTANCE_AUTO_TUNE="1",
                     DISTANCE_PID_DEADBAND_M="0.15")
    assert values == pytest.approx([distance, distance + 0.08, distance + 0.03, distance + 0.03, 0.03])


def test_manual_opt_out_keeps_explicit_profile():
    values = profile(TARGET_DISTANCE="1.6", FOLLOW_DISTANCE_AUTO_TUNE="0",
                     FOLLOW_FORWARD_START_DISTANCE_M="1.9", FOLLOW_FORWARD_STOP_DISTANCE_M="1.75",
                     FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_DISTANCE_M="1.65", DISTANCE_PID_DEADBAND_M="0.1")
    assert values == pytest.approx([1.6, 1.9, 1.75, 1.65, 0.1])
