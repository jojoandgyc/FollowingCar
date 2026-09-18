"""2.484 -> 2.502m is a continuous torso, not automatic surface reacquisition."""
import pytest
from test_astra_depth_consensus import sensor, CLIPPED


def test_continuous_multiregion_crosses_boundary_without_pending_or_clock_hold(sensor,caplog):
    caplog.set_level('INFO')
    runtime,clock,sample=sensor
    first=sample(2484)
    assert first.raw_distance_m is not None
    for value in (2502,2495,2502,2506):
        result=sample(value,advance=.05)
        assert result.raw_distance_m is not None
        assert result.sample_timestamp>first.sample_timestamp
        assert not result.detail.startswith('distance_jump_pending')
        assert result.jump_confirmation is None  # no forged re-anchor proof
    assert 'depth_boundary_continuity' in caplog.text


@pytest.mark.parametrize('change',['initial','uid','old','jump','small_previous','shift','area','weak_previous'])
def test_boundary_exception_never_bootstraps_or_bypasses_changed_evidence(sensor,change):
    runtime,clock,sample=sensor
    if change!='initial':sample(2300 if change=='small_previous' else 2484)
    if change=='weak_previous':runtime._last_accepted_region_count=1
    kwargs=dict(advance=.05)
    if change=='uid':kwargs['uid']=2
    if change=='old':kwargs['advance']=.3
    if change=='shift':kwargs['bbox']=(280.,20.,460.,479.)
    if change=='area':kwargs['bbox']=(220.,20.,440.,479.)
    result=sample(2700 if change=='jump' else 2502,**kwargs)
    assert result.raw_distance_m is None
    assert result.confirm_count>=1


def test_repeat_does_not_become_new_boundary_measurement(sensor):
    runtime,clock,sample=sensor
    sample(2484);result=sample(2502,advance=.05)
    again=runtime.measure_target(CLIPPED,640,480,target_id=1,use_latest_depth=True)
    assert again.raw_distance_m is None
    assert runtime._last_accepted_ts==result.sample_timestamp


def test_current_single_region_does_not_get_boundary_exception(sensor):
    import numpy as np
    runtime,clock,sample=sensor
    sample(2484)
    sparse=np.zeros((480,640),dtype=np.uint16)
    regions,_=runtime._torso_sampling_regions(CLIPPED,640,480,640,480)
    _,left,top,right,bottom=regions[-1]
    sparse[top:bottom,left:right]=2502
    result=sample(sparse,advance=.05)
    assert result.raw_distance_m is None
    assert result.jump_confirmation is None


def test_fast_boundary_change_still_requires_confirmation(sensor):
    runtime,clock,sample=sensor
    sample(2484)
    result=sample(2560,advance=.033)
    assert result.raw_distance_m is None
    assert result.confirm_count==1
