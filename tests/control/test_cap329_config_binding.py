import ast
import configparser
from pathlib import Path
from types import SimpleNamespace

from car_control_modular.config_loader import load_config_to_env


def test_real_ini_loader_and_main_binding(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr('os.environ', {})
    cfg = root/'car_control_modular/config/reid_runtime.ini'
    load_config_to_env(str(cfg))
    import os
    assert os.environ['FOLLOW_FORWARD_LOSS_HANDOFF_ENABLE'] == '1'
    assert os.environ['ASTRA_DEPTH_CONTINUATION_SPEED_CAP_ENABLE'] == '1'
    tree = ast.parse((root/'request_0513_modular.py').read_text())
    expressions = [k.value for n in ast.walk(tree) if isinstance(n, ast.Call)
                   for k in n.keywords if k.arg == 'follow_forward_loss_handoff_enable']
    assert len(expressions) == 1
    assert eval(compile(ast.Expression(expressions[0]), 'binding', 'eval'), {'os': os}) is True
    p = configparser.ConfigParser(); p.read(cfg)
    assert p['lateral_intent']['follow_cross_brake_mode'] == 'zero'
    assert float(p['astra_depth']['longitudinal_sample_max_age_sec']) == .30
