"""Exercise the real producer predicate, not just a permissive runtime stub."""
from dataclasses import replace
from types import SimpleNamespace
import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction, ControlDecision
from test_distance_tracking_response import setup
from test_distance_pi_controller import configured
from test_lateral_zero_runtime import owner


@pytest.mark.parametrize('case', ['valid', 'closing', 'unknown', 'near', 'old',
                                  'replay', 'quality', 'reverse', 'hazard'])
def test_only_fresh_outward_forward_evidence_can_interrupt_dwell(setup, owner, case):
    clock, c, frame = configured(setup)
    owner._follow_controller = c
    owner._near_yaw_park_request = SimpleNamespace(uid=1)
    owner.search_state = 'none'
    owner._last_command_capture_frame = 370
    c._braking_rate_source = 'raw_depth_window'
    c._braking_range_rate = .6
    f = frame(1.6, capture_timestamp=clock.now-.08, capture_frame_id=370)
    if case == 'closing': c._braking_range_rate = -.4
    if case == 'unknown': c._braking_rate_source = 'encoder_fallback'
    if case == 'near': f = frame(1.5)
    if case == 'old': f = frame(1.6, stamp=clock.now-.5)
    if case == 'replay': f = replace(f, distance_state=replace(f.distance_state, sample_timestamp=None))
    if case == 'hazard': f = replace(f, hazard=replace(f.hazard, active=True))
    action = ControlAction.backward(20, 'test') if case == 'reverse' else ControlAction.forward(40, 'test')
    decision = ControlDecision(actions=[action], reason='distance_pi')
    calls = []
    owner._release_near_yaw_park = lambda **kw: calls.append(kw) or False
    runtime.PersonTracker._release_near_yaw_park_for_decision(owner, f, f.persons[0], decision,
        control_source='vision', target_steerable=case != 'quality', low_quality_visible=case == 'quality')
    hints = [k['forward_resume_sample_ts'] for k in calls if k['forward_resume_sample_ts'] is not None]
    assert bool(hints) == (case == 'valid')
