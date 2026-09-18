import csv
import pytest
from tools.encoder_scale_audit import analyze, integrate_revolutions


def test_integration_uses_rpm_seconds_over_sixty_not_motor_commands():
    samples={100+i*.1:30. for i in range(11)}
    assert integrate_revolutions(samples,100.05,100.95)==pytest.approx(.45)


@pytest.mark.parametrize('samples,start,end',[
    ({100:30,100.2:30},100,100.2),
    ({100:30,100.2:30},100.05,100.1),
    ({100:30,100.1:30},99.9,100.1),
    ({100:30,100.1:30},100,100.2),
    ({100:30,100.1:float('nan')},100,100.1),
    ({100:-1,100.1:30},100,100.1),
    ({100:30,100.1:30},100.1,100),
])
def test_incomplete_or_invalid_encoder_data_cannot_produce_scale(samples,start,end):
    with pytest.raises(ValueError):integrate_revolutions(samples,start,end)


def fixture_run(tmp_path):
    with (tmp_path/'camera_raw.frames.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['capture_frame_id','capture_monotonic_sec'])
        writer.writeheader()
        writer.writerows([dict(capture_frame_id=1,capture_monotonic_sec=100),
                         dict(capture_frame_id=2,capture_monotonic_sec=101)])
    lines=[]
    for i in range(11):
        t=100+i*.1
        lines.append(f'visible_wheel_dispatch feedback_forward_rpm=(30, 30) feedback_ts={t}')
        if 0<i<10:
            lines.append(f'Astra depth timeline: target=1 sample_ts={t} temporal=new_sample raw={2-.3*i*.1} selected_regions=a+b')
    (tmp_path/'request_0513_modular.log').write_text('\n'.join(lines))
    return tmp_path


def test_depth_is_not_automatically_used_as_ground_truth(tmp_path):
    r=analyze(fixture_run(tmp_path),1,2)
    assert r['encoder_travel_m']==pytest.approx(.3)
    assert r['depth_comparison'] is None
    assert r['independently_implied_circumference_m'] is None
    assert r['calibration_applied'] is False


def test_independent_travel_and_depth_subinterval_are_separate(tmp_path):
    r=analyze(fixture_run(tmp_path),1,2,measured_travel=.35,fixed_target=True)
    assert r['independently_implied_circumference_m']==pytest.approx(.7)
    assert r['depth_comparison']['duration_sec']==pytest.approx(.8)
    assert r['depth_comparison']['depth_closure_m']==pytest.approx(.24)
    assert r['depth_comparison']['closure_to_encoder_ratio']==pytest.approx(1)
    assert r['calibration_applied'] is False
