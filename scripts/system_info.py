import platform
import socket
import psutil
import sys

def get_system_info() -> dict:
    uname = platform.uname()

    return {
        "hostname":    socket.gethostname(),
        "os":          f"{uname.system} {uname.release}",
        "cpu":         uname.processor or platform.processor(),
        "cpu_cores":   psutil.cpu_count(logical=False) or 1,
        "cpu_threads": psutil.cpu_count(logical=True)  or 1,
        "ram_gb":      round(psutil.virtual_memory().total / (1024**3), 2),
        "python":      sys.version.split()[0],
        "architecture": platform.machine(),
    }
