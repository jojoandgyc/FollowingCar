"""Exercise the real entry point with fake devices, including failure cleanup."""
import io
import logging

import pytest

import request_0513_modular as runtime
from car_control_modular.async_console_logging import BoundedAsyncConsoleHandler


@pytest.mark.parametrize("failure", [None, "init", "run"])
def test_main_installs_async_only_for_runtime_and_drains_after_cleanup(monkeypatch, failure):
    stream = io.StringIO()
    console = logging.StreamHandler(stream)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger = logging.Logger("fake-runtime", logging.INFO)
    logger.addHandler(console)
    events, handlers = [], []

    class Handler(BoundedAsyncConsoleHandler):
        def close(self):
            events.append("console_close")
            super().close()

    class Tracker:
        def __init__(self, model_path):
            assert model_path == "fake-model"
            assert console not in logger.handlers
            handler, = logger.handlers
            assert isinstance(handler, BoundedAsyncConsoleHandler)
            handlers.append(handler)
            logger.info("constructor")
            if failure == "init":
                raise RuntimeError("init failed")

        def run(self):
            try:
                logger.info("run")
                if failure == "run":
                    raise RuntimeError("run failed")
            finally:
                events.append("motor_cleanup")
                logger.info("motor cleanup finished")

    monkeypatch.setattr(runtime, "BoundedAsyncConsoleHandler", Handler)
    monkeypatch.setattr(runtime, "logger", logger)
    monkeypatch.setattr(runtime, "console_handler", console)
    monkeypatch.setattr(runtime, "PersonTracker", Tracker)
    monkeypatch.setattr(runtime.sys, "argv", ["runtime", "fake-model"])
    monkeypatch.setattr(runtime.signal, "signal", lambda *args: None)
    assert logger.handlers == [console], "importing does not start an output worker"
    if failure:
        with pytest.raises(RuntimeError, match=failure + " failed"):
            runtime.main()
    else:
        runtime.main()
    assert logger.handlers == [console]
    assert events == (["console_close"] if failure == "init" else ["motor_cleanup", "console_close"])
    assert handlers[0].stats()["pending"] == 0
    assert not handlers[0].stats()["worker_alive"]
    output = stream.getvalue()
    assert output.count("async_console_started") == 1
    assert "async_console_stopping" in output
    if failure != "init":
        assert output.index("motor cleanup finished") < output.index("async_console_stopping")


def test_import_keeps_console_synchronous():
    assert isinstance(runtime.console_handler, logging.StreamHandler)
    assert not any(isinstance(handler, BoundedAsyncConsoleHandler) for handler in runtime.logger.handlers)
