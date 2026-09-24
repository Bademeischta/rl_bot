"""Misst Durchsatz für verschiedene Thread-/Spiel-Aufteilungen mit der echten Trainings-Config.

    python tools/tune_threads.py --configs 16x64 16x128 8x128 16x256 --steps 2000000

Hintergrund: Auf der GPU macht jeder Sammel-Thread pro Schritt einen eigenen Inferenz-Aufruf
über seine Spiele. Die Aufruf-Latenz dominiert, nicht die Rechenlast, also senkt eine größere
Spielzahl pro Thread die Anzahl der Aufrufe und damit die Inferenzzeit.
Ergebnisse landen in bench/results/thread_tuning.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "build" / "cpp_cu128" / "train_bot.exe"
RESULTS = ROOT / "bench" / "results"


def load_metrics(path: Path, warmup: int) -> dict[str, float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))[warmup:]
    if not rows:
        return {}

    def mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if r.get(key)]
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "iterations": len(rows),
        "collected_sps": mean("Collected Steps/Second"),
        "overall_sps": mean("Overall Steps/Second"),
        "collect_s": mean("Collection Time"),
        "infer_s": mean("Policy Infer Time"),
        "sim_s": mean("Env Step Time"),
        "learn_s": mean("Consumption Time"),
        "iter_s": mean("Total Iteration Time"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=Path, default=ROOT / "train/configs/lucy_1v1.json")
    ap.add_argument("--configs", nargs="+", default=["16x64", "16x128", "8x128", "16x256"])
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    base = json.loads(a.base_config.read_text(encoding="utf-8"))
    py_home = subprocess.check_output([sys.executable, "-c", "import sys; print(sys.base_prefix)"],
                                      text=True).strip()
    env = dict(os.environ, PYTHONHOME=py_home, PATH=f"{py_home};{os.environ['PATH']}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []

    for spec in a.configs:
        threads, games = map(int, spec.split("x"))
        run_dir = ROOT / "runs" / f"tune_{spec}"
        shutil.rmtree(run_dir, ignore_errors=True)

        cfg = json.loads(json.dumps(base))
        cfg["learner"]["num_threads"] = threads
        cfg["learner"]["num_games_per_thread"] = games
        cfg["learner"]["checkpoint_folder"] = f"runs/tune_{spec}/checkpoints"
        cfg["learner"]["timesteps_per_save"] = 10 ** 12   # nicht speichern
        cfg_path = ROOT / f"runs/_tune_{spec}.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        print(f"\n=== {threads} Threads x {games} Spiele ({threads * games} Envs) ===", flush=True)
        start = time.time()
        log = run_dir.parent / f"tune_{spec}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as f:
            rc = subprocess.run([str(TRAINER), str(cfg_path), "--timestep-limit", str(a.steps)],
                                cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        wall = time.time() - start
        cfg_path.unlink(missing_ok=True)

        metrics = load_metrics(run_dir / "metrics.csv", a.warmup) if rc == 0 else {}
        if not metrics:
            print(f"  FEHLER (exit {rc}), siehe {log}")
            continue

        row = {"threads": threads, "games": games, "envs": threads * games,
               "wall_s": round(wall, 1),
               "effective_sps": round(a.steps / wall),
               **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in metrics.items()}}
        rows.append(row)
        print(f"  Inferenz {row['infer_s']:.2f}s  Sim {row['sim_s']:.2f}s  "
              f"Lernen {row['learn_s']:.2f}s  Iteration {row['iter_s']:.2f}s  "
              f"-> {row['overall_sps']:,.0f} SPS im Report", flush=True)

    if not rows:
        return
    out = RESULTS / "thread_tuning.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda r: r["overall_sps"])
    print(f"\nBeste Konfiguration: {best['threads']}x{best['games']} "
          f"mit {best['overall_sps']:,.0f} SPS (Report)")
    print(f"Geschrieben: {out}")


if __name__ == "__main__":
    main()
