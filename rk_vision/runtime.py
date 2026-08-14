from __future__ import annotations

import os
import tempfile
from typing import Any, List, Optional


class RKNNRuntimeError(RuntimeError):
    pass


class RKNNInferenceSession:
    """Small adapter over Rockchip RKNN Runtime.

    backend="auto" prefers rknn-toolkit-lite2 on the board and falls back to
    rknn-toolkit2 when running host-side tests.  backend="mock" is useful for
    import/logic tests without RKNN libraries.
    """

    def __init__(
        self,
        model_path: str,
        *,
        target: str = "rk3588",
        core_mask: str = "auto",
        backend: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.target = target
        self.core_mask = core_mask
        self.backend = backend
        self._rknn: Optional[Any] = None
        self._onnx_session: Optional[Any] = None
        self._onnx_input_name = ""
        self._onnx_input_shape: Optional[List[Any]] = None
        self._onnx_input_type = ""
        self._onnx_temp_path = ""
        self._loaded = False
        self._runtime_kind = ""

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        if self.backend == "mock":
            self._loaded = True
            self._runtime_kind = "mock"
            return
        if not self.model_path:
            raise FileNotFoundError("empty RKNN model path")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"RKNN model not found: {self.model_path}")

        if self.backend.lower().strip() in {"onnx", "onnxruntime", "ort"}:
            self._load_onnxruntime()
            self._loaded = True
            self._runtime_kind = "onnxruntime"
            return

        runtime_kind, klass = self._import_runtime()
        rknn = klass()
        ret = rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RKNNRuntimeError(f"load_rknn failed ret={ret}: {self.model_path}")

        if runtime_kind == "lite2":
            ret = self._init_lite_runtime(rknn)
        else:
            ret = self._init_toolkit_runtime(rknn)
        if ret != 0:
            raise RKNNRuntimeError(f"init_runtime failed ret={ret}: {self.model_path}")

        self._rknn = rknn
        self._loaded = True
        self._runtime_kind = runtime_kind

    def inference(self, inputs: List[Any], **kwargs: Any) -> List[Any]:
        self.load()
        if self.backend == "mock":
            return []
        if self._runtime_kind == "onnxruntime":
            return self._onnx_inference(inputs)
        if self._rknn is None:
            raise RKNNRuntimeError("RKNN runtime is not initialized")
        try:
            outputs = self._rknn.inference(inputs=inputs, **kwargs)
        except TypeError:
            outputs = self._rknn.inference(inputs=inputs)
        return [] if outputs is None else outputs

    def release(self) -> None:
        if self._rknn is not None:
            try:
                self._rknn.release()
            finally:
                self._rknn = None
        self._onnx_session = None
        if self._onnx_temp_path:
            try:
                os.unlink(self._onnx_temp_path)
            except OSError:
                pass
            self._onnx_temp_path = ""
        self._loaded = False

    def _import_runtime(self):
        requested = self.backend.lower().strip()
        if requested in {"auto", "rknnlite", "lite", "lite2"}:
            try:
                from rknnlite.api import RKNNLite

                return "lite2", RKNNLite
            except Exception as lite_exc:
                if requested not in {"auto"}:
                    raise RKNNRuntimeError("rknn-toolkit-lite2 is not available") from lite_exc
        if requested in {"auto", "rknn", "toolkit", "toolkit2"}:
            try:
                from rknn.api import RKNN

                return "toolkit2", RKNN
            except Exception as toolkit_exc:
                raise RKNNRuntimeError(
                    "No RKNN Python runtime found. Install rknn-toolkit-lite2 on RK3588 "
                    "or use backend='mock' for non-inference tests."
                ) from toolkit_exc
        raise ValueError(f"unknown RKNN backend: {self.backend!r}")

    def _load_onnxruntime(self) -> None:
        if not self.model_path.lower().endswith(".onnx"):
            raise ValueError("backend='onnxruntime' requires an ONNX model path")
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RKNNRuntimeError("onnxruntime is not available") from exc

        model_path = self._prepare_onnx_for_runtime(self.model_path)
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise RKNNRuntimeError(f"ONNXRuntime backend expects one input, got {len(inputs)}: {self.model_path}")
        inp = inputs[0]
        self._onnx_session = session
        self._onnx_input_name = str(inp.name)
        self._onnx_input_shape = list(inp.shape)
        self._onnx_input_type = str(inp.type)

    def _prepare_onnx_for_runtime(self, model_path: str) -> str:
        try:
            import onnx
        except Exception:
            return model_path
        model = onnx.load(model_path)
        if int(model.ir_version) <= 10:
            return model_path
        model.ir_version = 10
        fd, tmp_path = tempfile.mkstemp(prefix="rk_vision_", suffix=".onnx")
        os.close(fd)
        onnx.save(model, tmp_path)
        self._onnx_temp_path = tmp_path
        return tmp_path

    def _onnx_inference(self, inputs: List[Any]) -> List[Any]:
        if self._onnx_session is None:
            raise RKNNRuntimeError("ONNXRuntime session is not initialized")
        if len(inputs) != 1:
            raise ValueError(f"ONNXRuntime backend expects one input, got {len(inputs)}")
        array = self._prepare_onnx_input(inputs[0])
        outputs = self._onnx_session.run(None, {self._onnx_input_name: array})
        return [] if outputs is None else outputs

    def _prepare_onnx_input(self, value: Any) -> Any:
        np = _np()
        arr = np.asarray(value)
        expected = self._onnx_input_shape or []
        if arr.ndim == 4 and len(expected) == 4:
            channel_first = _static_dim(expected[1]) in {1, 3}
            input_is_nhwc = arr.shape[-1] in {1, 3}
            if channel_first and input_is_nhwc:
                arr = arr.transpose(0, 3, 1, 2)
        if "float" in self._onnx_input_type:
            if not np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype("float32") / 255.0
            else:
                arr = arr.astype("float32", copy=False)
        return np.ascontiguousarray(arr)

    def _init_lite_runtime(self, rknn: Any) -> int:
        mask = self._core_mask_value(rknn.__class__)
        if mask is None:
            return rknn.init_runtime()
        try:
            return rknn.init_runtime(core_mask=mask)
        except TypeError:
            return rknn.init_runtime()

    def _init_toolkit_runtime(self, rknn: Any) -> int:
        try:
            return rknn.init_runtime(target=self.target)
        except TypeError:
            return rknn.init_runtime()

    def _core_mask_value(self, runtime_class: Any) -> Optional[int]:
        name = str(self.core_mask or "auto").strip().lower()
        mapping = {
            "auto": ("NPU_CORE_AUTO", "NPU_CORE_0_1_2"),
            "all": ("NPU_CORE_0_1_2", "NPU_CORE_AUTO"),
            "0": ("NPU_CORE_0",),
            "1": ("NPU_CORE_1",),
            "2": ("NPU_CORE_2",),
            "0_1": ("NPU_CORE_0_1",),
            "1_2": ("NPU_CORE_1_2",),
            "0_1_2": ("NPU_CORE_0_1_2",),
        }
        for attr in mapping.get(name, ()):
            if hasattr(runtime_class, attr):
                return int(getattr(runtime_class, attr))
        return None


def _static_dim(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _np():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy is required for runtime input adaptation") from exc
    return np
