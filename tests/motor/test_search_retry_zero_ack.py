"""Successful serial-write acknowledgement, entirely fake backend."""
import pytest
import car_control_modular.action_runtime as action_module
from test_depth_drive_rpm import make_runtime


@pytest.mark.parametrize("method", ["drive", "turn"])
def test_zero_ack_is_after_write_and_not_renewed(method, monkeypatch):
    motion, owner, _driver, _symbols = make_runtime()
    owner._search_retry_zero_requested_at = 10.
    owner._search_retry_zero_sent_at = None
    monkeypatch.setattr(action_module.time, "monotonic", lambda: 10.02)
    send = (lambda: motion.send_percent_drive(0)) if method == "drive" else motion.send_rotate_pulse_zero_stop
    send()
    assert owner._search_retry_zero_sent_at == 10.02
    monkeypatch.setattr(action_module.time, "monotonic", lambda: 10.12)
    send()
    assert owner._search_retry_zero_sent_at == 10.02


@pytest.mark.parametrize("method", ["drive", "turn"])
def test_failed_zero_cannot_acknowledge_observation(method):
    motion, owner, _driver, _symbols = make_runtime()
    backend = motion.backend
    owner._search_retry_zero_requested_at = 10.
    owner._search_retry_zero_sent_at = None
    def fail(*args, **kwargs): raise OSError("fake motor failure")
    if method == "drive": backend.send_diff = fail
    else: backend.send_targets = fail
    with pytest.raises(OSError):
        motion.send_percent_drive(0) if method == "drive" else motion.send_rotate_pulse_zero_stop()
    assert owner._search_retry_zero_sent_at is None
