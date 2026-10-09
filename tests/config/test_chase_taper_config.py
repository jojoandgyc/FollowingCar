"""Validate the real config/import binding without constructing hardware."""
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest
from car_control_modular.config_loader import load_config_to_env

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'car_control_modular/config/reid_runtime.ini'


def test_runtime_taper_is_explicit_and_legacy_config_stays_compatible(monkeypatch, tmp_path):
    env = {}
    monkeypatch.setattr(os, 'environ', env)
    load_config_to_env(str(CONFIG))
    assert env['DISTANCE_PI_LAUNCH_FULL_ERROR_M'] == '0.5'
    old = tmp_path / 'old.ini'
    old.write_text('[distance_pid]\ncontrol_mode=distance_pi\npi_launch_request_rpm=180\n')
    env.clear()
    load_config_to_env(str(old))
    assert 'DISTANCE_PI_LAUNCH_FULL_ERROR_M' not in env


@pytest.mark.parametrize('value', ['nan', 'inf', '-1', '2.1'])
def test_main_rejects_invalid_value_before_hardware(value):
    env = dict(os.environ, DISTANCE_PI_LAUNCH_FULL_ERROR_M=value)
    p = subprocess.run([sys.executable, '-c', 'import request_0513_modular'], cwd=ROOT,
                       env=env, capture_output=True, text=True)
    assert p.returncode != 0
    assert 'DISTANCE_PI_LAUNCH_FULL_ERROR_M must be finite and within 0..2' in p.stderr


def test_main_config_and_actual_follow_config_constructor_binding():
    p = subprocess.run([sys.executable, '-c',
        'import request_0513_modular as r; assert r.DISTANCE_PI_LAUNCH_FULL_ERROR_M == .5',
        '--config', str(CONFIG)], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    tree = ast.parse((ROOT / 'request_0513_modular.py').read_text())
    bindings = [k.value.id for n in ast.walk(tree) if isinstance(n, ast.Call)
                for k in n.keywords if k.arg == 'distance_pi_launch_full_error_m'
                and isinstance(k.value, ast.Name)]
    assert bindings == ['DISTANCE_PI_LAUNCH_FULL_ERROR_M']
