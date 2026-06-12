import json
import os
import platform
import sys
from datetime import datetime

from scripts.bench_cpu        import run_cpu_benchmarks
from scripts.bench_memory     import run_memory_benchmarks
from scripts.bench_workloads  import run_workload_benchmarks
from scripts.system_info      import get_system_info

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def banner():
    print(f"""
{CYAN}{BOLD}
╔══════════════════════════════════════════════╗
║         HARDWARE BENCHMARK SUITE             ║
║   Proiect: Computer Architecture Analysis    ║
╚══════════════════════════════════════════════╝
{RESET}""")

def section(title):
    print(f"\n{YELLOW}{BOLD}{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}{RESET}")

def ok(msg):
    print(f"{GREEN}  ✓ {msg}{RESET}")

def main():
    banner()

    section("PASUL 1/4 — Colectare informații sistem")
    sys_info = get_system_info()
    ok(f"CPU:       {sys_info['cpu']}")
    ok(f"RAM:       {sys_info['ram_gb']} GB")
    ok(f"OS:        {sys_info['os']}")
    ok(f"Python:    {sys_info['python']}")
    ok(f"CPU cores: {sys_info['cpu_cores']} (logice: {sys_info['cpu_threads']})")

    section("PASUL 2/4 — Benchmark CPU (throughput integer & float)")
    cpu_results = run_cpu_benchmarks()
    for name, val in cpu_results.items():
        ok(f"{name}: {val:.4f}s")

    section("PASUL 3/4 — Benchmark Memorie & Cache")
    mem_results = run_memory_benchmarks()
    for name, val in mem_results.items():
        ok(f"{name}: {val:.4f}s")

    section("PASUL 4/4 — Workload-uri reale (sortare, join, ML, SQL)")
    wl_results = run_workload_benchmarks()
    for name, val in wl_results.items():
        ok(f"{name}: {val:.4f}s")

    section("SALVARE REZULTATE")
    all_results = {
        "timestamp":    datetime.now().isoformat(),
        "system_info":  sys_info,
        "cpu":          cpu_results,
        "memory":       mem_results,
        "workloads":    wl_results,
    }

    os.makedirs("results", exist_ok=True)
    out_file = f"results/results_{sys_info['hostname']}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    ok(f"Rezultate salvate în: {out_file}")

if __name__ == "__main__":
    main()
