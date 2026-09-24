"""Phase-0 C++-SPS-Sweep (RLGymPPO_CPP, Binary aus bench/cpp/build.ps1).

python bench/cpp_sps.py --flavor cpu   --device cpu  --configs 16x24 16x32 12x32
python bench/cpp_sps.py --flavor cu128 --device cuda --configs 16x24

Rohdaten pro Iteration: bench/results/cpp_sps_raw.csv
Mittelwerte (ohne Warmup) pro Konfiguration: bench/results/cpp_sps.csv
"""
import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from python_sps import ResourceSampler  # noqa: E402

RESULTS = ROOT / "bench" / "results"
RAW_FIELDS = ["device", "threads", "games", "itr", "cdl", "collected_sps", "overall_sps",
              "collect_time_s", "consume_time_s", "cumulative_ts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flavor", choices=["cpu", "cu128"], default="cpu")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--configs", nargs="+", default=["16x24"], help="<threads>x<gamesPerThread>")
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--collect-during-learn", action="store_true",
                    help="Weiter Steps sammeln, während der Learner rechnet (hilft vor allem auf CPU)")
    a = ap.parse_args()

    build = ROOT / "build" / f"cpp_{a.flavor}"
    exe = build / "bench_cpp_sps.exe"
    py_home = subprocess.check_output([sys.executable, "-c", "import sys; print(sys.base_prefix)"],
                                      text=True).strip()
    env = dict(os.environ, PYTHONHOME=py_home, PATH=f"{py_home};{os.environ['PATH']}")
    RESULTS.mkdir(parents=True, exist_ok=True)

    for cfg in a.configs:
        threads, games = map(int, cfg.split("x"))
        raw = RESULTS / f"_raw_{a.flavor}_{cfg}.csv"
        raw.unlink(missing_ok=True)
        cdl = "1" if a.collect_during_learn else "0"
        log = RESULTS / f"cpp_sps_{a.flavor}_{a.device}_{cfg}_cdl{cdl}{a.tag}.log"
        print(f"\n=== {a.flavor}/{a.device} threads={threads} games/thread={games} ===", flush=True)

        sampler = ResourceSampler()
        sampler.start()
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run([str(exe), a.device, str(threads), str(games), str(a.steps),
                                 str(raw), str(ROOT / "collision_meshes"), cdl],
                                cwd=build, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        sampler.stop()
        if rc != 0 or not raw.exists():
            print(f"  FEHLER (exit {rc}), siehe {log}")
            continue

        rows = [dict(zip(RAW_FIELDS, r)) for r in csv.reader(raw.open(encoding="utf-8"))]
        with open(RESULTS / "cpp_sps_raw.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([[a.flavor + a.tag] + list(r.values()) for r in rows])
        raw.unlink()
        kept = rows[a.warmup:] or rows

        def avg(k):
            return round(sum(float(r[k]) for r in kept) / len(kept), 2)

        res = {"flavor": a.flavor + a.tag, "device": a.device, "threads": threads, "games": games,
               "collect_during_learn": int(cdl),
               "iterations": len(kept), "collected_sps": int(avg("collected_sps")),
               "overall_sps": int(avg("overall_sps")), "collect_time_s": avg("collect_time_s"),
               "consume_time_s": avg("consume_time_s"), "cpu_pct": sampler.mean("cpu"),
               "ram_used_gb": sampler.mean("ram_used_gb"), "gpu_util_pct": sampler.mean("gpu_util"),
               "timestamp": time.strftime("%Y-%m-%d %H:%M")}
        print("  ", res, flush=True)
        out = RESULTS / "cpp_sps.csv"
        new = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(res))
            if new:
                w.writeheader()
            w.writerow(res)


if __name__ == "__main__":
    main()
