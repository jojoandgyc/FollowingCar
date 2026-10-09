"""Fresh comparable distance can replace an uncertainty-only memory stop.

This is not target-speed matching or permission to reuse an expired motor
grant. The new request cannot exceed measured RPM or the previous approval.
"""
import pytest

from test_cap224_uncertain_zero_settling import ORIGIN, RECOVER, ZERO, controller, observe


def ready(*, old_target=.11230466670656515, approved=36.):
    c = controller()
    first = observe(c, ORIGIN, age=.177, distance=1.7552, raw=1.759,
                    ego=0., rate=old_target, target=old_target, rise_rpm_per_sec=0.)
    if first.output_rpm > approved:
        c.accept_output_limit(ORIGIN, approved)
    c.suspend(40125.029, 'lateral_zero_no_qualified_depth:visual_pid_center_hold',
              reset_execution=True)
    return c


def fallback(c, **changes):
    args = dict(stamp=ZERO, age=.0376, distance=1.759, raw=1.7698,
                ego=8.5, rate=None, valid=False, evidence=False,
                allow_motion_memory_endpoint_fallback=True)
    args.update(changes)
    return observe(c, **args)


def test_cap222_farther_raw_endpoint_avoids_first_uncertainty_zero():
    c = ready()
    original_memory = c._motion_memory
    r = fallback(c)
    assert r.memory_endpoint_fallback
    assert r.brake_source == 'raw_endpoint_distance_bound'
    assert r.output_rpm == 8
    assert r.cap_rpm == 8.5
    assert r.output_rpm <= r.memory_endpoint_cap_rpm
    assert r.memory_endpoint_cap_rpm > 30
    assert r.effective_range_rate_m_s == pytest.approx(-.21420672027)
    assert r.target_velocity_bound_m_s < 0  # No invented person-speed credit.
    assert r.motion_origin_ts == ORIGIN
    assert c._motion_memory == original_memory  # Not renewed by the new fallback.
    assert not r.motion_window_used
    assert not c._uncertain_zero_brake_anchor
    assert c._last_sample_ts == ZERO  # NEW physical distance, not an old lease.
    second = observe(c)
    third = observe(c, 40125.222647697, age=.038, raw=1.7552, ego=7.,
                    rate=-.12032371479001751, target=-.013394546866453121)
    assert (r.output_rpm, second.output_rpm, third.output_rpm) == (8, 22, 37)
    assert not second.brake_settling_limited and not third.brake_settling_limited


@pytest.mark.parametrize('case', ['default_flag', 'geometry_unqualified', 'marginal_rotation',
                                 'old_target_approaching', 'raw_closer', 'high_closure',
                                 'high_ego', 'reverse_ego', 'missing_ego', 'zero_ego',
                                 'no_previous_approval'])
def test_unqualified_or_real_closing_motion_cannot_use_endpoint_fallback(case):
    c = ready(old_target=-.14 if case == 'old_target_approaching' else .11230466670656515,
              approved=0. if case == 'no_previous_approval' else 36.)
    args = {}
    if case in {'default_flag', 'geometry_unqualified'}:
        args['allow_motion_memory_endpoint_fallback'] = False
    if case == 'marginal_rotation': args['motion_memory_rotation_bound'] = .35
    if case == 'raw_closer': args['raw'] = 1.748
    if case == 'high_closure': args.update(raw=1.55, distance=1.55)
    if case == 'high_ego': args['ego'] = 80.
    if case == 'reverse_ego': args['ego'] = -1.
    if case == 'missing_ego': args['ego'] = None
    if case == 'zero_ego': args['ego'] = 0.
    r = fallback(c, **args)
    assert not r.memory_endpoint_fallback
    if case != 'missing_ego':
        assert r.output_rpm == 0


@pytest.mark.parametrize('kind', ['new_near_raw', 'near_filtered'])
def test_actual_near_distance_always_wins(kind):
    c = ready()
    r = fallback(c, **({'raw': 1.41} if kind == 'new_near_raw' else {'distance': 1.41}))
    assert not r.memory_endpoint_fallback and r.output_rpm == 0


@pytest.mark.parametrize('kind', ['duplicate', 'older', 'new_too_late', 'new_expired',
                                'memory_expired', 'raw_jump'])
def test_endpoint_fallback_never_refreshes_missing_or_expired_evidence(kind):
    c = ready()
    args = {}
    if kind == 'duplicate': args.update(stamp=ORIGIN, age=.18)
    if kind == 'older': args.update(stamp=ORIGIN-.01, age=.18)
    if kind == 'new_too_late': args.update(age=.181)
    if kind == 'new_expired': args.update(age=.251)
    if kind == 'memory_expired': args.update(stamp=ORIGIN+.36, age=.01)
    if kind == 'raw_jump': args['measurement_jump_clamped'] = True
    before = c._last_sample_ts
    r = fallback(c, **args)
    assert not r.memory_endpoint_fallback
    if kind != 'memory_expired':
        assert c._last_sample_ts == before


def test_fallback_cannot_raise_previous_lower_approved_command():
    c = ready(approved=4.)
    r = fallback(c)
    assert r.memory_endpoint_fallback
    assert r.output_rpm == r.cap_rpm == 4.


def test_memory_fallback_cannot_replace_an_already_measured_braking_maneuver():
    c = controller()
    first = observe(c, ORIGIN, age=.02, distance=1.7552, raw=1.759,
                    ego=100., rate=-1.2, target=.16, rise_rpm_per_sec=0.)
    assert first.cap_rpm < 95.
    c.suspend(40125.029, 'physical_depth_expired', reset_execution=True)
    r = fallback(c)
    assert not r.memory_endpoint_fallback and r.output_rpm == 0


@pytest.mark.parametrize('kind', ['geometry_rejected', 'request_rejected', 'parking'])
def test_actual_faults_do_not_become_endpoint_distance_permission(kind):
    c = ready()
    if kind == 'geometry_rejected': c.invalidate_motion_memory()
    if kind == 'request_rejected': c.reject_output(ORIGIN)
    if kind == 'parking': c.set_normal_parking(True)
    r = fallback(c)
    assert not r.memory_endpoint_fallback
