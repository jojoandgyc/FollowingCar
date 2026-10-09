"""Read-only numeric-library audit; never changes a live process's thread pools."""
import ctypes
import json
import os


def numeric_runtime_info():
    import numpy as np
    result = {
        "numpy_version": np.__version__, "numpy_path": np.__file__,
        "environment": {key: os.environ.get(key) for key in (
            "FOLLOW_NUMERIC_THREADS", "FOLLOW_KALMAN_SOLVER", "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")},
        "libraries": [],
    }
    try:
        with open("/proc/self/maps", encoding="utf-8") as stream:
            paths = sorted({line.split(maxsplit=5)[-1].strip() for line in stream
                            if "/" in line and any(name in line.lower()
                            for name in ("openblas", "libmkl_rt", "libblis"))})
        for path in paths:
            item = {"path": path, "actual_threads": None}
            try:
                lib = ctypes.CDLL(path, mode=getattr(os, "RTLD_NOLOAD", 0))
                for symbol in ("openblas_get_num_threads64_", "openblas_get_num_threads",
                               "scipy_openblas_get_num_threads", "MKL_Get_Max_Threads"):
                    getter = getattr(lib, symbol, None)
                    if getter is not None:
                        getter.argtypes, getter.restype = [], ctypes.c_int
                        item["actual_threads"] = int(getter())
                        item["getter"] = symbol
                        break
            except (OSError, AttributeError) as exc:
                item["error"] = type(exc).__name__
            result["libraries"].append(item)
    except OSError as exc:
        result["audit_error"] = type(exc).__name__
    return result


def log_numeric_runtime(logger, phase):
    # Observability failures must not interrupt tracking or safety handling.
    try:
        logger.info("numeric_runtime phase=%s %s", phase,
                    json.dumps(numeric_runtime_info(), separators=(",", ":")))
    except Exception as exc:
        logger.warning("numeric_runtime_audit_failed phase=%s error=%s", phase, type(exc).__name__)
