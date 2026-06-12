import time
import numpy as np

N_REPEAT = 3

def _time_it(func, *args) -> float:
    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    return min(times)

def _sequential_read():
    arr = np.ones(32_000_000, dtype=np.float64)
    total = 0.0
    for _ in range(4):
        total += arr.sum()
    return total

def _random_access():
    size = 8_000_000
    arr  = np.arange(size, dtype=np.float64)
    idx  = np.random.randint(0, size, size=500_000)
    return arr[idx].sum()

def _cache_test(size_bytes: int) -> float:
    n_elements = size_bytes // 8
    arr = np.ones(n_elements, dtype=np.float64)
    n_ops = max(1, 64_000_000 // n_elements)

    t0 = time.perf_counter()
    for _ in range(n_ops):
        arr += 1.0
    return time.perf_counter() - t0


def _cache_l1():   return _cache_test(32   * 1024)
def _cache_l2():   return _cache_test(256  * 1024)
def _cache_l3():   return _cache_test(8    * 1024 * 1024)
def _cache_ram():  return _cache_test(64   * 1024 * 1024)

def _write_bandwidth():
    arr = np.empty(32_000_000, dtype=np.float64)
    for _ in range(4):
        arr[:] = 3.14
    return arr.sum()

def run_memory_benchmarks() -> dict:
    return {
        "mem_sequential_read_s":  _time_it(_sequential_read),
        "mem_random_access_s":    _time_it(_random_access),
        "mem_write_bandwidth_s":  _time_it(_write_bandwidth),
        "cache_l1_s":             _cache_l1(),
        "cache_l2_s":             _cache_l2(),
        "cache_l3_s":             _cache_l3(),
        "cache_ram_s":            _cache_ram(),
    }
