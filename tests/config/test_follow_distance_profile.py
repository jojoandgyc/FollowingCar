"""Only import configuration: never construct PersonTracker or start hardware."""
import json
import os
import configparser
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def profile(config=None, **overrides):
    env = dict(os.environ, **overrides)
    result = subprocess.run(
        [sys.executable, "-c", "import json; import request_0513_modular as r; "
         "print(json.dumps([r.TARGET_DISTANCE,r.FOLLOW_FORWARD_START_DISTANCE_M,"
         "r.FOLLOW_FORWARD_STOP_DISTANCE_M,r.FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_DISTANCE_M,"
         "r.DISTANCE_PID_DEADBAND_M]))",
         *(["--config", str(config)] if config is not None else [])],
        cwd=ROOT, env=env, check=True, text=True, capture_output=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("distance", [1.3, 1.5, 1.6, 2.0])
def test_one_distance_setting_derives_small_hysteresis(distance):
    values = profile(TARGET_DISTANCE=str(distance), FOLLOW_DISTANCE_AUTO_TUNE="1",
                     DISTANCE_PID_DEADBAND_M="0.15")
    assert values == pytest.approx([distance, distance + 0.08, distance + 0.03, distance + 0.03, 0.03])


def test_manual_opt_out_keeps_explicit_profile():
    values = profile(TARGET_DISTANCE="1.6", FOLLOW_DISTANCE_AUTO_TUNE="0",
                     FOLLOW_FORWARD_START_DISTANCE_M="1.9", FOLLOW_FORWARD_STOP_DISTANCE_M="1.75",
                     FOLLOW_NEAR_DISTANCE_ROTATE_ONLY_DISTANCE_M="1.65", DISTANCE_PID_DEADBAND_M="0.1")
    assert values == pytest.approx([1.6, 1.9, 1.75, 1.65, 0.1])


def test_default_launcher_uses_follow_ini_and_derives_its_distance():
    # Follow the real launcher selection, not an environment-only surrogate.
    # Importing the entrypoint loads --config but never constructs hardware.
    launcher = (ROOT / "run_request_0428_modular.sh").read_text()
    default = re.search(r'^DEFAULT_CONFIG="([^"]+)"$', launcher, re.MULTILINE)
    assert default is not None
    assert default.group(1) == "car_control_modular/config/reid_runtime.ini"
    config = ROOT / default.group(1)
    parser = configparser.ConfigParser()
    assert parser.read(config, encoding="utf-8")
    assert not parser.getboolean("follow", "rotation_only", fallback=False)
    assert parser.get("distance", "source") == "vision_depth"
    assert parser.getboolean("distance", "auto_tune_follow_distance")
    distance = parser.getfloat("distance", "target_distance_m")
    # The selected INI must override inherited values and stale manual limits.
    values = profile(config=config, TARGET_DISTANCE="9.9", FOLLOW_DISTANCE_AUTO_TUNE="0",
                     FOLLOW_FORWARD_START_DISTANCE_M="9.98",
                     FOLLOW_FORWARD_STOP_DISTANCE_M="9.93")
    assert values == pytest.approx(
        [distance, distance + 0.08, distance + 0.03, distance + 0.03, 0.03]
    )
