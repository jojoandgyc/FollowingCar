"""No hardware: a withdrawn raw search turn must never become percent RPM."""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_control_modular.action_runtime import MotionActionRuntime


class FakeBackend:
    def __init__(self):
        self.config = SimpleNamespace(m1_is_left_wheel=True)
        self.targets = []
        self.percent_writes = []
        self.motion_armed = False

    @staticmethod
    def wheel_raw_state_to_target(_side, raw, state):
        return int(raw) if state == 0x01 else -int(raw) if state == 0x02 else 0

    @staticmethod
    def clip_percent(value):
        return max(0, min(100, int(value)))

    def send_targets(self, left, right, label):
        self.targets.append((left, right, label))
        self.motion_armed = bool(left or right)

    def send_diff(self, left, left_state, right, right_state, label):
        self.percent_writes.append((left, right, label))
        # Reproduce the reported fallback: 10 percent means 20 RPM here.
        self.send_targets(
            self.wheel_raw_state_to_target("left", 2 * left, left_state),
            self.wheel_raw_state_to_target("right", 2 * right, right_state),
            label,
        )


def make_runtime(*, raw=8, source="search"):
    symbols = SimpleNamespace(
        forward=0, backward=1, rotate_left=2, rotate_right=3,
        stop=4, steer_left=5, steer_right=6,
        action_names={2: "rotate_left", 3: "rotate_right"},
    )
    owner = SimpleNamespace(
        motor_io_lock=threading.Lock(),
        command_lock=threading.Lock(),
        current_command=symbols.rotate_left,
        command_start_time=time.time(),
        frame_index=88,
        _current_rotate_raw_target=raw,
        _current_rotate_raw_source=source,
        _current_rotate_turn_percent=10,
        _current_rotate_pulse_enabled=False,
        _last_motor_dispatch_source="keepalive_refresh",
    )
    config = SimpleNamespace(
        steering_feedback_median_window=3,
        motor_rotate_raw_target=18,
        rotate_turn_percent_from_forward=10,
        rotate_pulse_brake_enable=False,
        rotate_duration=0.18,
        rotate_pulse_pause_sec=0.0,
        rotate_hold_stale_sec=0.25,
    )
    backend = FakeBackend()
    runtime = MotionActionRuntime(
        owner, backend, config, symbols,
        hard_stop_check=lambda _action=None: False,
        logger=logging.getLogger("search-zero-fallback"),
    )
    return runtime, owner, backend, symbols


@pytest.mark.parametrize(
    "source",
    [
        "search", "candidate_approach", "reacquire_depth_wait",
        "search_candidate_evidence_observe", "confirmed_search_reacquire_hold",
        "parked_pid_encoder", "lateral_intent_30hz", "lateral_intent_revoked",
        "stale_vision_zero_yaw", "future_explicit_raw_source",
    ],
)
@pytest.mark.parametrize("side", ["rotate_left", "rotate_right"])
def test_explicit_raw_zero_never_uses_percent_fallback(source, side):
    runtime, owner, backend, symbols = make_runtime(raw=0, source=source)
    runtime.send_robot_command(getattr(symbols, side))
    assert backend.targets == []
    assert backend.percent_writes == []
    assert owner._current_rotate_turn_percent == 10
    assert not hasattr(owner, "_last_motor_dispatch_ts")


@pytest.mark.parametrize("source", ["default", "", None])
def test_legacy_percent_mode_still_works_without_an_explicit_raw_source(source):
    runtime, owner, backend, symbols = make_runtime(raw=0, source=source)
    if source is None:
        del owner._current_rotate_raw_source
    runtime.send_robot_command(symbols.rotate_left)
    assert backend.percent_writes == [(10, 10, "TURN")]
    assert backend.targets == [(-20, 20, "TURN")]


@pytest.mark.parametrize("source", ["search", "reacquire_depth_wait", "parked_pid_encoder"])
def test_nonzero_explicit_raw_turn_keeps_its_exact_rpm(source):
    runtime, _owner, backend, symbols = make_runtime(raw=8, source=source)
    runtime.send_robot_command(symbols.rotate_left)
    assert backend.targets == [(-8, 8, "TURN")]
    assert backend.percent_writes == []


class PausePreparedPacket:
    """Let observation zero overtake a TURN waiting to enter motor I/O."""

    def __init__(self):
        self.prepared = threading.Event()
        self.resume = threading.Event()
        self.lock = threading.Lock()

    def __enter__(self):
        if threading.current_thread().name == "prepared-old-turn":
            self.prepared.set()
            assert self.resume.wait(2.0), "test never released the prepared packet"
        self.lock.acquire()

    def __exit__(self, *_args):
        self.lock.release()


def start_prepared_turn(runtime, owner, action):
    gate = PausePreparedPacket()
    owner.motor_io_lock = gate
    worker = threading.Thread(
        target=lambda: runtime.send_robot_command(action),
        name="prepared-old-turn", daemon=True,
    )
    worker.start()
    assert gate.prepared.wait(1.0), "TURN did not reach the motor lock"
    return gate, worker


def finish_prepared_turn(gate, worker):
    gate.resume.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()


@pytest.mark.parametrize("use_publisher_revision", [False, True])
def test_observation_zero_cancels_old_packet_even_after_same_search_is_rearmed(
    use_publisher_revision,
):
    runtime, owner, backend, symbols = make_runtime()
    if use_publisher_revision:
        owner._lateral_yaw_revision = 10
    gate, worker = start_prepared_turn(runtime, owner, symbols.rotate_left)
    try:
        owner._current_rotate_raw_target = 0
        owner._current_rotate_turn_percent = 0
        owner._current_rotate_raw_source = "search_candidate_evidence_observe"
        if use_publisher_revision:
            owner._lateral_yaw_revision += 1
        assert runtime.cancel_active_rotate_for_observation("candidate_observe")
        assert backend.targets == [(0, 0, "TURN_ZERO")]

        # Even restoring the exact source and RPM does not authorize the
        # packet prepared before cancellation. Only a new dispatch may turn.
        owner._current_rotate_raw_target = 8
        owner._current_rotate_raw_source = "search"
        owner.current_command = symbols.rotate_left
    finally:
        finish_prepared_turn(gate, worker)
    assert backend.targets == [(0, 0, "TURN_ZERO")]
    assert backend.percent_writes == []
    assert not hasattr(owner, "_last_motor_dispatch_ts")
    runtime.send_robot_command(symbols.rotate_left)
    assert backend.targets[-1] == (-8, 8, "TURN")


def test_raw_zero_is_rechecked_at_motor_lock_before_revision_commit():
    runtime, owner, backend, symbols = make_runtime()
    gate, worker = start_prepared_turn(runtime, owner, symbols.rotate_left)
    try:
        # A producer can be part-way through its zero fields before it commits
        # the revision. The old raw packet must already respect raw zero.
        owner._current_rotate_raw_target = 0
    finally:
        finish_prepared_turn(gate, worker)
    assert backend.targets == []
    assert backend.percent_writes == []


def test_prepared_legacy_percent_packet_cannot_override_explicit_observation_zero():
    runtime, owner, backend, symbols = make_runtime(raw=0, source="default")
    gate, worker = start_prepared_turn(runtime, owner, symbols.rotate_left)
    try:
        owner._current_rotate_raw_source = "search_candidate_evidence_observe"
        backend.send_targets(0, 0, "OBSERVATION_ZERO")
    finally:
        finish_prepared_turn(gate, worker)
    assert backend.targets == [(0, 0, "OBSERVATION_ZERO")]
    assert backend.percent_writes == []


def test_search_packet_honors_publisher_yaw_revision():
    runtime, owner, backend, symbols = make_runtime()
    owner._lateral_yaw_revision = 4
    gate, worker = start_prepared_turn(runtime, owner, symbols.rotate_left)
    try:
        # Isolate the publisher contract from source/raw and runtime-cancel
        # checks: a revoked and rearmed identical search still gets a new token.
        owner._lateral_yaw_revision = 5
        backend.send_targets(0, 0, "OBSERVATION_ZERO")
    finally:
        finish_prepared_turn(gate, worker)
    assert backend.targets == [(0, 0, "OBSERVATION_ZERO")]


def test_old_raw_magnitude_is_not_replayed_after_a_new_search_speed():
    runtime, owner, backend, symbols = make_runtime(raw=8)
    gate, worker = start_prepared_turn(runtime, owner, symbols.rotate_left)
    try:
        owner._current_rotate_raw_target = 4
    finally:
        finish_prepared_turn(gate, worker)
    assert backend.targets == []
    runtime.send_robot_command(symbols.rotate_left)
    assert backend.targets == [(-4, 4, "TURN")]
