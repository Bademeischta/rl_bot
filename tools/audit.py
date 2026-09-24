"""Phase-0 Hardware-Audit: CPU/RAM/Disk + RTX-5070-(sm_120)-Check. Schreibt docs/audit.json."""
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def cpu_info():
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        flags = set(info.get("flags", []))
        brand = info.get("brand_raw", platform.processor())
    except Exception:
        flags, brand = set(), platform.processor()
    return {
        "brand": brand,
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "avx2": "avx2" in flags,
        "avx512f": "avx512f" in flags,
    }


def gpu_info():
    import torch
    out = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
           "cuda_available": torch.cuda.is_available()}
    if not out["cuda_available"]:
        return out
    out["name"] = torch.cuda.get_device_name(0)
    out["capability"] = list(torch.cuda.get_device_capability(0))
    out["arch_list"] = torch.cuda.get_arch_list()
    out["cudnn"] = torch.backends.cudnn.version()
    out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    try:
        a = torch.randn(4096, 4096, device="cuda")
        b = a @ a  # Warmup: cuBLAS-Init nicht mitmessen
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            b = a @ a
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        out["matmul_ok"] = bool(torch.isfinite(b).all())
        out["matmul_tflops_fp32"] = round(20 * 2 * 4096**3 / dt / 1e12, 1)
    except RuntimeError as e:  # z.B. "no kernel image is available"
        out["matmul_ok"] = False
        out["matmul_error"] = str(e)[:300]
    return out


def main():
    disk = shutil.disk_usage(ROOT)
    report = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": sys.version.split()[0],
        "cpu": cpu_info(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "disk_free_gb": round(disk.free / 1024**3, 1),
        "repo_in_onedrive": "onedrive" in str(ROOT).lower(),
        "gpu": gpu_info(),
    }
    g = report["gpu"]
    checks = {
        "cuda_available": g.get("cuda_available", False),
        "sm_120": g.get("capability") == [12, 0],
        "sm_120_in_torch_build": "sm_120" in g.get("arch_list", []),
        "gpu_matmul": g.get("matmul_ok", False),
        "avx2": report["cpu"]["avx2"],
        "ram_ge_16gb": report["ram_gb"] >= 15,
        "not_onedrive": not report["repo_in_onedrive"],
    }
    report["checks"] = checks
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print()
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
