"""Low-overhead wall/thread CPU timing; never controls tracking decisions."""
import time


class StageTiming:
    def __init__(self):
        self.started = self.previous = time.perf_counter()
        self.cpu_started = self.cpu_previous = time.thread_time()
        self.values = {}

    def mark(self, name):
        wall, cpu = time.perf_counter(), time.thread_time()
        self.values[name] = (wall - self.previous) * 1000.0
        self.values[name + "_cpu"] = (cpu - self.cpu_previous) * 1000.0
        self.previous, self.cpu_previous = wall, cpu

    def finish(self):
        self.values["total"] = (time.perf_counter() - self.started) * 1000.0
        self.values["cpu"] = (time.thread_time() - self.cpu_started) * 1000.0
        # Not a pure lock-wait metric: native workers and OS scheduling also
        # contribute. Thread CPU intentionally excludes other runtime threads.
        self.values["noncpu"] = max(0.0, self.values["total"] - self.values["cpu"])
        return self.values
