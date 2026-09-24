"""Phase-0 Python-SPS-Benchmark (rlgym-ppo + RLGym v2 + RocketSim).

Sweep:   python bench/python_sps.py --n-proc 8 16 24 32 --steps 6000000
Einzeln: python bench/python_sps.py --single 16 --steps 6000000

Jede Konfiguration läuft in einem eigenen Subprozess. Die ersten --warmup Iterationen
werden verworfen, danach gehen Mittelwerte über alle restlichen Iterationen in
bench/results/python_sps.csv (eine Zeile pro n_proc).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "bench" / "results"

NETS = {"small": (256, 256, 256), "large": (2048, 2048, 1024, 1024)}


class ResourceSampler(threading.Thread):
    """Sampelt CPU-%, RAM der Env-Kindprozesse und GPU-Auslastung alle `period` s."""

    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        import psutil
        self.psutil = psutil
        self.period = period
        self.samples = []
        self._stop = threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml, self.gpu = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nvml = self.gpu = None

    def run(self):
        me = self.psutil.Process()
        self.psutil.cpu_percent(None)
        while not self._stop.wait(self.period):
            kids = me.children(recursive=True)
            rss = []
            for k in kids:
                try:
                    rss.append(k.memory_info().rss)
                except self.psutil.Error:
                    pass
            s = {"cpu": self.psutil.cpu_percent(None),
                 "n_children": len(rss),
                 "child_rss_mb": (sum(rss) / len(rss) / 2**20) if rss else 0.0,
                 "ram_used_gb": self.psutil.virtual_memory().used / 2**30}
            if self.gpu is not None:
                s["gpu_util"] = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu).gpu
            self.samples.append(s)

    def stop(self):
        self._stop.set()

    def mean(self, key):
        v = [s[key] for s in self.samples if key in s]
        return round(sum(v) / len(v), 1) if v else None


def run_single(n_proc, steps, warmup, net, device, out_json):
    from rlgym_ppo import Learner
    from rlgym_ppo.util import reporting
    from env.factory import build_default_env

    reports = []
    orig = reporting.report_metrics

    def capture(loggable_metrics, debug_metrics, wandb_run=None):
        reports.append(dict(loggable_metrics))
        orig(loggable_metrics, debug_metrics, wandb_run)

    reporting.report_metrics = capture

    tmp_ckpt = ROOT / "runs" / "bench_tmp"
    sampler = ResourceSampler()
    learner = Learner(
        build_default_env,
        n_proc=n_proc,
        min_inference_size=max(1, int(round(n_proc * 0.9))),
        timestep_limit=steps,
        ts_per_iteration=100_000,
        exp_buffer_size=300_000,
        ppo_batch_size=100_000,
        ppo_minibatch_size=50_000,
        ppo_epochs=1,
        policy_layer_sizes=NETS[net],
        critic_layer_sizes=NETS[net],
        standardize_returns=True,
        standardize_obs=False,
        log_to_wandb=False,
        checkpoints_save_folder=str(tmp_ckpt),
        checkpoint_load_folder=None,
        save_every_ts=10**12,
        device=device,
    )
    sampler.start()
    t0 = time.time()
    learner.learn()
    wall = time.time() - t0
    sampler.stop()

    kept = reports[warmup:] or reports

    def avg(k):
        v = [r[k] for r in kept if k in r]
        return round(sum(v) / len(v), 1) if v else None

    result = {
        "n_proc": n_proc, "net": net, "device": device, "iterations": len(kept),
        "collected_sps": avg("Collected Steps per Second"),
        "overall_sps": avg("Overall Steps per Second"),
        "collect_time_s": avg("Timestep Collection Time"),
        "consume_time_s": avg("Timestep Consumption Time"),
        "cpu_pct": sampler.mean("cpu"),
        "rss_per_proc_mb": sampler.mean("child_rss_mb"),
        "ram_used_gb": sampler.mean("ram_used_gb"),
        "gpu_util_pct": sampler.mean("gpu_util"),
        "wall_s": round(wall, 1),
    }
    Path(out_json).write_text(json.dumps(result), encoding="utf-8")
    os._exit(0)  # rlgym-ppo-Kindprozesse nicht auf sauberes Shutdown warten lassen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, nargs="+", default=[8, 16, 24, 32])
    ap.add_argument("--single", type=int)
    ap.add_argument("--steps", type=int, default=6_000_000)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--net", choices=NETS, default="small")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json")
    a = ap.parse_args()

    if a.single is not None:
        run_single(a.single, a.steps, a.warmup, a.net, a.device, a.out_json)
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "python_sps.csv"
    new = not csv_path.exists()
    for n in a.n_proc:
        out_json = RESULTS / f"_tmp_py_{n}.json"
        out_json.unlink(missing_ok=True)
        print(f"\n=== n_proc={n} net={a.net} steps={a.steps:,} ===", flush=True)
        log = RESULTS / f"python_sps_n{n}_{a.net}.log"
        with open(log, "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, __file__, "--single", str(n), "--steps", str(a.steps),
                            "--warmup", str(a.warmup), "--net", a.net, "--device", a.device,
                            "--out-json", str(out_json)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if not out_json.exists():
            print(f"  FEHLER, siehe {log}")
            continue
        r = json.loads(out_json.read_text(encoding="utf-8"))
        out_json.unlink()
        r["timestamp"] = time.strftime("%Y-%m-%d %H:%M")
        print("  ", r, flush=True)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(r))
            if new:
                w.writeheader()
                new = False
            w.writerow(r)


if __name__ == "__main__":
    main()
