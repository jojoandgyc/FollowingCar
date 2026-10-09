"""Execution-continuity investigation: classify evidence loss, not relax it.

The CAP386 soft ROI gap was already implemented. CAP1192's larger rotation
uncertainty is deliberately not converted into the same memory permission.
No motor backend or camera is started by these tests.
"""
from dataclasses import replace

import pytest

from test_distance_tracking_response import setup
from test_distance_pi_controller import step
from test_cap386_geometry_memory import primed, current


@pytest.mark.parametrize('changes, detail', [
    (dict(distance=2.16, x=.165, yaw=-25.08, raw_yaw=-18.81, age=.15),
     'rotation_uncertainty_limit'),
    (dict(age=.31), 'geometry_age_limit'),
    (dict(age=.15, yaw=36., raw_yaw=36.), 'yaw_limit'),
])
def test_rejection_distinguishes_age_yaw_and_rotation_uncertainty(setup, caplog, changes, detail):
    clock, controller, frame = primed(setup)
    assert controller._distance_pid._distance_pi._motion_memory is not None
    caplog.clear()
    step(controller, current(frame, clock, **changes))
    records = [r.message for r in caplog.records if r.message.startswith('distance_closure_rejected ')]
    assert len(records) == 1
    assert f'detail={detail} ' in records[0]
    assert 'sample_inserted=False' in records[0]
    assert 'deadline_renewed=False' in records[0]
    assert 'rotation_uncertainty_limit_m_s=0.25' in records[0]
    assert not controller._raw_closing_window.samples
    assert controller._distance_pid._distance_pi._motion_memory is None
    assert controller.last_distance_pid_result.pi_brake_source != 'relative_motion_memory'


def test_cap1192_rejection_reports_numeric_uncertainty_not_stale_roi(setup, caplog):
    clock, controller, frame = primed(setup)
    caplog.clear()
    step(controller, current(frame, clock, distance=2.16, x=.165,
                             yaw=-25.08, raw_yaw=-18.81, age=.15))
    record = next(r.message for r in caplog.records
                  if r.message.startswith('distance_closure_rejected '))
    fields = dict(part.split('=', 1) for part in record.split() if '=' in part)
    assert float(fields['rotation_uncertainty_m_s']) == pytest.approx(.4390781563)
    assert float(fields['geometry_age_ms']) == pytest.approx(150.)
    assert fields['detail'] == 'rotation_uncertainty_limit'


def test_soft_roi_gap_keeps_original_memory_clock_and_cannot_accelerate(setup, caplog):
    clock, controller, frame = primed(setup)
    origin = controller._raw_closing_window.samples[-1][0]
    raw_window = list(controller._raw_closing_window.samples)
    previous = controller.last_distance_pid_result.output_rpm
    for elapsed in (.05, .10, .15):
        clock.now = origin+elapsed
        caplog.clear()
        step(controller, current(frame, clock))
        result = controller.last_distance_pid_result
        assert controller._raw_closing_window.samples == raw_window
        assert controller._braking_rate_source != 'raw_depth_window'
        assert result.pi_brake_source == 'relative_motion_memory'
        assert result.pi_motion_origin_ts == origin
        assert result.output_rpm <= previous
        previous = result.output_rpm
        record = next(r.message for r in caplog.records
                      if r.message.startswith('distance_closure_skipped '))
        fields = dict(part.split('=', 1) for part in record.split() if '=' in part)
        assert float(fields['memory_remaining_ms']) == pytest.approx((.18-elapsed)*1000.)
        assert fields['sample_inserted'] == fields['deadline_renewed'] == 'False'
    clock.now = origin+.181
    step(controller, current(frame, clock))
    assert not controller._raw_closing_window.samples
    assert controller._distance_pid._distance_pi._motion_memory is None


def test_soft_gap_diagnostics_do_not_mask_immediate_hazard(setup, caplog):
    clock, controller, frame = primed(setup)
    caplog.clear()
    observation = current(frame, clock)
    observation = replace(observation, hazard=replace(observation.hazard, active=True))
    decision = step(controller, observation)
    assert not controller._raw_closing_window.samples
    assert controller._distance_pid._distance_pi._motion_memory is None
    assert decision.current_forward_percent == 0
    assert 'distance_closure_skipped' not in caplog.text
