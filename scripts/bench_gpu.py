import time
import numpy as np

try:
    import cupy as cp
    _gpu_available = True
    _gpu_name = cp.cuda.Device(0).attributes.get("DeviceName", "Unknown GPU")
    _gpu_vram_gb = round(
        cp.cuda.Device(0).mem_info[1] / (1024 ** 3), 2
    )
except Exception:
    cp = None
    _gpu_available = False
    _gpu_name = "N/A"
    _gpu_vram_gb = 0.0

BACKEND = "cuda" if _gpu_available else "cpu_fallback"
N_REPEAT = 3
MATRIX_SIZE = 4096
ARRAY_SIZE  = 50_000_000

def _sync():
    if _gpu_available:
        cp.cuda.Stream.null.synchronize()


def _time_it(func, *args) -> float:
    times = []
    for _ in range(N_REPEAT):
        _sync()
        t0 = time.perf_counter()
        func(*args)
        _sync()
        times.append(time.perf_counter() - t0)
    return min(times)

def _matrix_multiply():
    if _gpu_available:
        xp = cp
    else:
        xp = np
    rng = np.random.default_rng(42)
    A = xp.array(rng.random((MATRIX_SIZE, MATRIX_SIZE)).astype(np.float32))
    B = xp.array(rng.random((MATRIX_SIZE, MATRIX_SIZE)).astype(np.float32))
    C = A @ B
    return C


def _elementwise_ops():
    if _gpu_available:
        xp = cp
    else:
        xp = np
    rng = np.random.default_rng(42)
    a = xp.array(rng.random(ARRAY_SIZE).astype(np.float32))
    result = xp.sqrt(a) + xp.exp(a * 0.001) + xp.sin(a)
    return result


def _memory_transfer_host_to_device():
    if not _gpu_available:
        a = np.ones(32_000_000, dtype=np.float32)
        b = a.copy()
        return b
    a_cpu = np.ones(32_000_000, dtype=np.float32)   # 128 MB
    a_gpu = cp.array(a_cpu)
    return a_gpu


def _memory_transfer_device_to_host():
    if not _gpu_available:
        a = np.ones(32_000_000, dtype=np.float32)
        b = a.copy()
        return b
    a_gpu = cp.ones(32_000_000, dtype=np.float32)
    a_cpu = cp.asnumpy(a_gpu)
    return a_cpu


def _multi_stream_ops():
    chunk = ARRAY_SIZE // 4
    rng = np.random.default_rng(42)

    if _gpu_available:
        streams = [cp.cuda.Stream() for _ in range(4)]
        arrays  = [cp.array(rng.random(chunk).astype(np.float32)) for _ in range(4)]
        results = []
        for i, (stream, arr) in enumerate(zip(streams, arrays)):
            with stream:
                results.append(cp.sqrt(arr) + cp.sin(arr))
        for s in streams:
            s.synchronize()
    else:
        arrays = [np.array(rng.random(chunk).astype(np.float32)) for _ in range(4)]
        results = [np.sqrt(a) + np.sin(a) for a in arrays]

    return results

def run_gpu_benchmarks() -> dict:
    results = {
        "gpu_backend":         BACKEND,
        "gpu_name":            _gpu_name,
        "gpu_vram_gb":         _gpu_vram_gb,
    }

    results["gpu_matmul_s"]           = _time_it(_matrix_multiply)
    results["gpu_elementwise_s"]      = _time_it(_elementwise_ops)
    results["gpu_h2d_transfer_s"]     = _time_it(_memory_transfer_host_to_device)
    results["gpu_d2h_transfer_s"]     = _time_it(_memory_transfer_device_to_host)
    results["gpu_multistream_s"]      = _time_it(_multi_stream_ops)

    return results
