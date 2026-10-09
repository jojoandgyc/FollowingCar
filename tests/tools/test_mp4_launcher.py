"""Launcher post-processing tests with stub runtime only; never use devices."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not shutil.which("flock"), reason="Linux flock required")


def workspace(tmp_path):
    shutil.copy2(ROOT / "run_request_0428_modular.sh", tmp_path)
    (tmp_path / "tools").mkdir()
    (tmp_path / "car_control_modular/config").mkdir(parents=True)
    (tmp_path / "car_control_modular/config/reid_runtime.ini").write_text("[test]\n")
    (tmp_path / "tools/prepare_run_logs.py").write_text('''
from pathlib import Path
import sys
directory = Path(sys.argv[1]) / "run_stub"
directory.mkdir(parents=True)
print(directory)
''')
    (tmp_path / "request_0513_modular.py").write_text('''
from pathlib import Path
import os
import sys
directory = Path(os.environ["FOLLOW_LOG_DIR"])
(directory / "camera_raw.avi").write_bytes(b"stub video")
(directory / "runtime.pid").write_text(str(os.getpid()))
if os.environ.get("STUB_CLOSED", "1") == "1":
    print("Camera recording closed: submitted=3 written=3 dropped=0")
if os.environ.get("STUB_WRITER_ERROR") == "1":
    print("Camera recorder finalize failed: writer stopped")
sys.exit(int(os.environ.get("STUB_RC", "0")))
''')
    (tmp_path / "tools/export_recording_mp4.py").write_text('''
from pathlib import Path
import os
import sys
directory = Path(sys.argv[1]).parent
pid = int((directory / "runtime.pid").read_text())
try:
    os.kill(pid, 0)
except ProcessLookupError:
    pass
else:
    raise AssertionError("export overlapped runtime")
(directory / "export.called").write_text("after runtime exit")
sys.exit(int(os.environ.get("STUB_EXPORT_RC", "0")))
''')
    return tmp_path / "run_request_0428_modular.sh"


@pytest.mark.parametrize("env,called,rc", [
    ({}, True, 0),
    ({"FOLLOW_VIDEO_EXPORT_MP4": "0"}, False, 0),
    ({"STUB_RC": "7"}, False, 7),
    ({"STUB_CLOSED": "0"}, False, 0),
    ({"STUB_WRITER_ERROR": "1"}, False, 0),
    ({"STUB_EXPORT_RC": "1"}, True, 0),
])
def test_export_is_post_exit_optional_and_does_not_replace_runtime_rc(tmp_path, env, called, rc):
    launcher = workspace(tmp_path)
    settings = {**os.environ, "PY": sys.executable, "PERIPHERAL_PREFLIGHT": "0",
                "REQUEST_LOOPBACK_UP": "0", "FOLLOW_VIDEO_EXPORT_MP4": "1", **env}
    result = subprocess.run(["sh", str(launcher)], env=settings, capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == rc, result.stdout + result.stderr
    directory = tmp_path / "run_request_0428_modular_logs/run_stub"
    assert (directory / "export.called").exists() is called
    assert (directory / "camera_raw.avi").read_bytes() == b"stub video"


def test_session_lock_blocks_relaunch_before_log_rotation(tmp_path):
    import fcntl
    launcher = workspace(tmp_path)
    with (tmp_path / ".follow_session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["sh", str(launcher)], capture_output=True, text=True,
                                env={**os.environ, "PY": sys.executable}, timeout=10)
    assert result.returncode == 5
    assert not (tmp_path / "run_request_0428_modular_logs").exists()


def test_session_lock_symlink_is_not_followed_or_truncated(tmp_path):
    launcher = workspace(tmp_path)
    target = tmp_path / "user_data"
    target.write_text("keep")
    (tmp_path / ".follow_session.lock").symlink_to(target)
    result = subprocess.run(["sh", str(launcher)], capture_output=True, text=True,
                            env={**os.environ, "PY": sys.executable}, timeout=10)
    assert result.returncode == 5
    assert target.read_text() == "keep"
