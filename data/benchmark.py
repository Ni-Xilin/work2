"""Measure repeated model/generator calls without adding a profiling dependency."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import importlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np


def _resolve_callable(spec: str):
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("callable must use module:function syntax")
    value = importlib.import_module(module_name)
    for part in attribute.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"resolved object is not callable: {spec}")
    return value


def _rss_mb() -> float:
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_memory = psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        if not get_memory(handle, ctypes.byref(counters), counters.cb):
            return float("nan")
        return float(counters.WorkingSetSize / (1024**2))
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def benchmark(function, *, repetitions: int, warmup: int, items_per_call: int, sample_interval: float) -> dict[str, np.ndarray]:
    if repetitions <= 0 or warmup < 0 or items_per_call <= 0 or sample_interval <= 0:
        raise ValueError("repetitions/items-per-call/sample-interval must be positive and warmup non-negative")
    for _ in range(warmup):
        function()
    latency = np.empty(repetitions, dtype=np.float64)
    throughput = np.empty(repetitions, dtype=np.float64)
    cpu = np.empty(repetitions, dtype=np.float64)
    peak_rss = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        stop = threading.Event()
        memory_samples = [_rss_mb()]

        def sample_memory() -> None:
            while not stop.wait(sample_interval):
                memory_samples.append(_rss_mb())

        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        function()
        elapsed = time.perf_counter() - wall_start
        cpu_elapsed = time.process_time() - cpu_start
        stop.set()
        sampler.join()
        memory_samples.append(_rss_mb())
        latency[index] = elapsed * 1000.0
        throughput[index] = items_per_call / max(elapsed, 1e-15)
        cpu[index] = cpu_elapsed / max(elapsed, 1e-15) * 100.0
        finite_memory = [value for value in memory_samples if np.isfinite(value)]
        peak_rss[index] = max(finite_memory) if finite_memory else np.nan
    return {
        "latency_ms": latency,
        "throughput_per_second": throughput,
        "cpu_percent": cpu,
        "peak_ram_mb": peak_rss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a zero-argument callable or callable factory.")
    parser.add_argument("--callable", dest="callable_spec", help="Zero-argument callable as module:function.")
    parser.add_argument("--factory", help="Factory as module:function; its return value is benchmarked.")
    parser.add_argument("--factory-kwargs", default="{}", help="JSON keyword arguments passed to the factory.")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--items-per-call", type=int, default=1)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.callable_spec) == bool(args.factory):
        raise ValueError("provide exactly one of --callable and --factory")
    if args.factory:
        function = _resolve_callable(args.factory)(**json.loads(args.factory_kwargs))
        if not callable(function):
            raise TypeError("factory must return a callable")
    else:
        function = _resolve_callable(args.callable_spec)
    measurements = benchmark(
        function,
        repetitions=args.repetitions,
        warmup=args.warmup,
        items_per_call=args.items_per_call,
        sample_interval=args.sample_interval,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **measurements)
    print(f"benchmark measurements: {args.output}")


if __name__ == "__main__":
    main()
