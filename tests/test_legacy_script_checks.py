"""Collect the reviewed, hardware-free script checks in the normal suite.

Do not discover arbitrary main() functions: tests/motor/test_mssd_live.py and
tools/ contain live utilities. Explicit subprocesses isolate INI/environment,
fake HAL module injection, clocks and global monkey patches from other tests.
"""
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKS = (
    ('config/test_rotation_only_config.py', ()),
    ('config/test_rk3588_runtime_config.py', ()),
    ('control/test_search_candidate_gate.py', ()),
    ('control/test_vision_depth_control.py', ()),
    ('control/test_lateral_intent.py', ()),
    ('control/test_controller_logic.py', ()),
    ('control/test_hazard_runtime.py', ()),
    ('control/test_steering_pid.py', ()),
    ('control/test_distance_fusion.py', ()),
    ('control/test_sensor_runtime.py', ()),
    ('control/test_distance_runtime_filter.py', ()),
    ('control/test_distance_pid.py', ()),
    ('control/test_search_status_boundary.py', ()),
    ('sensors/test_side_ir_filter.py', ()),
    ('sensors/test_astra_depth_runtime.py', ()),
    ('sensors/test_mmwave_health.py', ()),
    ('sensors/test_ir_iio.py', ('--fake',)),
    ('sensors/test_ultrasonic_iio.py', ('--fake',)),
    ('vision/test_video_recorder.py', ()),
    ('vision/test_yolo11_two_class_output.py', ()),
    ('vision/test_predicted_duplicate_suppression.py', ()),
    ('vision/test_gstreamer_capture.py', ()),
    ('vision/test_reid_appearance_fusion.py', ()),
    ('vision/test_identity_bank.py', ()),
    ('vision/test_search_frame_quality.py', ()),
    ('vision/test_search_detector_probe.py', ()),
    ('vision/test_deepsort_tracker.py', ()),
    ('motor/test_reverse_transition.py', ('--config', 'car_control_modular/config/reid_runtime.ini')),
    ('motor/test_rotation_only_pulse_runtime.py', ()),
    ('motor/test_mssd_mapping.py', ()),
)


@pytest.mark.parametrize('script,args', CHECKS, ids=[row[0] for row in CHECKS])
def test_safe_script(script, args):
    # Runtime imports in other tests load INIs into os.environ. Do not let
    # that cross-test state silently select a different mode in a subprocess.
    env = {key: value for key, value in os.environ.items() if not key.startswith((
        'FOLLOW_', 'DISTANCE_', 'VISION_', 'VISIBLE_', 'Y8_', 'RKNN_',
        'MOTOR_', 'ASTRA_', 'MMWAVE_', 'MODULE_', 'TARGET_', 'TRACK_',
        'STEER_', 'ROTATE_', 'ROTATION_', 'LATERAL_', 'SEARCH_', 'IR_',
        'ULTRASONIC_', 'IMU_', 'BUNKER_', 'SIDE_IR_', 'SAFETY_',
        'DEPTH_', 'HISTORICAL_', 'DIRECT_STOP_',
    ))}
    env['PYTHONPATH'] = str(ROOT)
    result = subprocess.run([sys.executable, str(ROOT/'tests'/script), *args],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_script_only_check_is_silently_omitted():
    script_only = set()
    for path in (ROOT/'tests').rglob('test_*.py'):
        module = ast.parse(path.read_text())
        functions = [node.name for node in module.body if isinstance(node, ast.FunctionDef)]
        collected_class = any(isinstance(node, ast.ClassDef) and (
            node.name.startswith('Test') or any(isinstance(base, ast.Attribute)
                and base.attr == 'TestCase' for base in node.bases)) for node in module.body)
        if 'main' in functions and not collected_class and not any(
                name.startswith('test_') for name in functions):
            script_only.add(path.relative_to(ROOT/'tests').as_posix())
    # Explicitly NOT auto-executed, even if a future edit changes its defaults.
    assert script_only == {row[0] for row in CHECKS} | {'motor/test_mssd_live.py'}


def test_config_expected_dictionary_has_no_silently_overwritten_keys():
    path = ROOT/'tests/config/test_rk3588_runtime_config.py'
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Dict):
            names = [key.value for key in node.keys if isinstance(key, ast.Constant)
                     and isinstance(key.value, str)]
            assert len(names) == len(set(names)), 'duplicate config expectation'
