#!/usr/bin/env python3
"""Offline numerical timing only: no models, camera, serial or motor imports.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/benchmark_kalman.py
Select BLAS threads BEFORE launching Python; this tool never changes live pools.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--solver", choices=("numpy", "triangular", "both"), default="both")
    args = parser.parse_args()
    if not 10 <= args.iterations <= 10000:
        parser.error("iterations must be between 10 and 10000")
    import numpy as np
    from rk_vision.deepsort.kalman_filter import KalmanFilter
    from rk_vision.numeric_runtime import numeric_runtime_info
    result = {"note": "offline microbenchmark, not live control performance", "runs": {}}
    outputs = {}
    for solver in (("numpy", "triangular") if args.solver == "both" else (args.solver,)):
        kf = KalmanFilter(solver=solver)
        z = np.array([320., 240., .5, 400.])
        mean, covariance = kf.initiate(z)
        measurement = z + [2., 1., .01, -1.]
        for _ in range(10): kf.update(mean, covariance, measurement)
        kf.timing = {}
        wall, cpu = [], []
        for _ in range(args.iterations):
            t, c = time.perf_counter(), time.thread_time()
            outputs[solver] = kf.update(mean, covariance, measurement)
            cpu.append((time.thread_time()-c)*1000.)
            wall.append((time.perf_counter()-t)*1000.)
        result["runs"][solver] = {
            "iterations": args.iterations, "mean_ms": float(np.mean(wall)),
            "p50_ms": float(np.percentile(wall, 50)), "p95_ms": float(np.percentile(wall, 95)),
            "p99_ms": float(np.percentile(wall, 99)), "max_ms": float(max(wall)),
            "thread_cpu_mean_ms": float(np.mean(cpu)), "stage_totals": kf.timing,
        }
    if len(outputs) == 2:
        for a, b in zip(outputs["numpy"], outputs["triangular"]):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)
        result["numerically_equivalent"] = True
    result["runtime"] = numeric_runtime_info()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
