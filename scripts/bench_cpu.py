import time
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

N_SMALL  = 5_000_000
N_LARGE  = 50_000_000
N_REPEAT = 3


def _time_it(func, *args) -> float:
    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return min(times)

def _integer_ops():
    a = np.arange(N_LARGE, dtype=np.int64)
    b = np.arange(N_LARGE, dtype=np.int64) + 1
    c = a * b + a % 1000
    return c.sum()

def _float_ops():
    a = np.random.rand(N_LARGE).astype(np.float64)
    b = np.random.rand(N_LARGE).astype(np.float64)
    c = np.sqrt(a * b + a)
    return c.sum()

def _worker_task(_):
    a = np.random.rand(N_SMALL).astype(np.float64)
    return np.dot(a, a)

def _multithread_ops():
    n_threads = multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(executor.map(_worker_task, range(n_threads)))
    return results

def run_cpu_benchmarks() -> dict:
    return {
        "cpu_integer_throughput_s": _time_it(_integer_ops),
        "cpu_float_throughput_s":   _time_it(_float_ops),
        "cpu_multithread_s":        _time_it(_multithread_ops),
    }
