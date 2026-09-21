import json
from pathlib import Path
import numpy as np
import pytest
from tools import audit_reid_crops as audit
from rk_vision.reid import _fuse_appearance_features


def test_split_distances_reconstruct_fused_distance():
    a=np.array([1.,0.]);b=np.array([.8,.6])
    c=np.array([0.,1.]);d=np.array([.6,.8]);w=.35
    fused=audit.cosine(_fuse_appearance_features(a,c,w),_fuse_appearance_features(b,d,w))
    assert fused==pytest.approx((audit.cosine(a,b)+w*w*audit.cosine(c,d))/(1+w*w),abs=1e-6)

@pytest.mark.parametrize("value",[np.zeros(2),np.array([np.nan,1]),np.ones(3)])
def test_invalid_embeddings_fail_explicitly(value):
    with pytest.raises(ValueError):audit.cosine(value,np.ones(2))


def test_process_guard_finds_car_not_shell_command_text(tmp_path):
    for pid,args in [(1,b"python3\0/x/request_0513_modular.py\0"),
                     (2,b"bash\0-lc\0pgrep request_0513_modular.py\0")]:
        p=tmp_path/str(pid);p.mkdir();(p/"cmdline").write_bytes(args)
    assert audit.car_processes(tmp_path)==[1]


def test_crop_selection_refuses_ambiguity(tmp_path):
    p=tmp_path/"reid_diagnostics";p.mkdir()
    for n in (1,2):(p/f"frame_capture_00000001_track_{n}.png").touch()
    with pytest.raises(ValueError):audit.crop_path(tmp_path,1)


def test_default_audit_never_runs_inference(tmp_path,monkeypatch,capsys):
    event={"frame_index":1,"track_id":1,"output_uid":0,"reason":"wait","bank_updated":False,
           "query_metadata":{"capture_frame_id":1}}
    (tmp_path/"request_0513_modular.log").write_text("reid_match_evidence "+json.dumps(event)+"\n")
    monkeypatch.setattr(audit,"extract_saved",lambda *a:pytest.fail("unexpected inference"))
    assert audit.main(["--run-dir",str(tmp_path),"--templates","1","--queries","1"])==0
    report=json.loads(capsys.readouterr().out)
    assert report["mode"]=="log_only"


def test_report_never_overwrites_existing_file(tmp_path):
    p=tmp_path/"report.json";p.write_text("keep")
    with pytest.raises(SystemExit):audit.main(["--run-dir",str(tmp_path),"--templates","1","--queries","1","--output",str(p)])
    assert p.read_text()=="keep"


def test_inference_refuses_running_car(tmp_path,monkeypatch):
    (tmp_path/"request_0513_modular.log").write_text("")
    monkeypatch.setattr(audit,"car_processes",lambda:[123])
    monkeypatch.setattr(audit,"extract_saved",lambda *a:pytest.fail("unexpected inference"))
    with pytest.raises(SystemExit):audit.main(["--run-dir",str(tmp_path),"--templates","1","--queries","1","--infer"])
