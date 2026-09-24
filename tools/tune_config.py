"""Vergleicht Trainings-Konfigurationen fair: jede Variante startet vom selben Checkpoint.

    python tools/tune_config.py --steps 2000000

Warum vom selben Checkpoint: Ein frischer Lauf ist deutlich langsamer, weil die Policy den Ball
kaum trifft und die Episoden ständig in den Timeout laufen (siehe docs/phase0_results.md).
Varianten von null zu starten würde also vor allem die Anlaufphase messen.

Ergebnisse: bench/results/config_tuning.csv
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
sys.path.insert(0, str(ROOT / "tools"))
from show_metrics import read_rows  # noqa: E402

TRAINER = ROOT / "build" / "cpp_cu128" / "train_bot.exe"
RESULTS = ROOT / "bench" / "results"

# Name -> Überschreibungen in der Config. Leerer Patch = unveränderte Ausgangskonfiguration.
VARIANTS: dict[str, dict] = {
    # Config unverändert übernehmen (für Vergleiche zwischen Binaries)
    "aktuell": {},
    "baseline": {"metrics": {"skill_update_interval": 4}},
    "ohne_skilltracker": {"metrics": {"skill_tracker": False}},
    "iteration_200k": {"learner": {"timesteps_per_iteration": 200000, "ppo_batch_size": 200000}},
    "ppo_epochs_1": {"learner": {"ppo_epochs": 1}},
    "skill_intervall_16": {"metrics": {"skill_update_interval": 16}},
    "skill_intervall_32": {"metrics": {"skill_update_interval": 32}},
}


def deep_merge(base: dict, patch: dict) -> dict:
    out = json.loads(json.dumps(base))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def latest_checkpoint(run_dir: Path) -> Path:
    dirs = [d for d in (run_dir / "checkpoints").iterdir()
            if d.is_dir() and d.name.isdigit() and (d / "PPO_POLICY.lt").exists()]
    if not dirs:
        raise SystemExit(f"Keine Checkpoints in {run_dir}")
    return max(dirs, key=lambda d: int(d.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=Path, default=ROOT / "train/configs/lucy_1v1.json")
    ap.add_argument("--seed-run", type=Path, default=ROOT / "runs/lucy_1v1")
    ap.add_argument("--steps", type=int, default=2_000_000, help="Steps pro Variante")
    ap.add_argument("--warmup", type=int, default=3, help="Iterationen, die verworfen werden")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--exe", type=Path, default=TRAINER)
    ap.add_argument("--tag", default="")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Wiederholungen je Variante. Die Streuung zwischen identischen Läufen "
                         "liegt bei etwa 4 Prozent, Einzelmessungen taugen daher nichts.")
    a = ap.parse_args()

    base = json.loads(a.base_config.read_text(encoding="utf-8"))
    seed = latest_checkpoint(a.seed_run)
    seed_steps = int(seed.name)
    print(f"Ausgangs-Checkpoint: {seed} ({seed_steps:,} Steps)")

    py_home = subprocess.check_output([sys.executable, "-c", "import sys; print(sys.base_prefix)"],
                                      text=True).strip()
    env = dict(os.environ, PYTHONHOME=py_home, PATH=f"{py_home};{os.environ['PATH']}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []

    plan = [(name, rep) for name in a.variants for rep in range(a.repeats)]
    for name, rep in plan:
        if name not in VARIANTS:
            raise SystemExit(f"Unbekannte Variante: {name}")
        run_dir = ROOT / "runs" / f"tune_{name}{a.tag}_{rep}"
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.copytree(seed, run_dir / "checkpoints" / str(seed_steps))

        cfg = deep_merge(base, VARIANTS[name])
        cfg["learner"]["checkpoint_folder"] = f"runs/tune_{name}{a.tag}_{rep}/checkpoints"
        cfg["learner"]["timesteps_per_save"] = 10 ** 12
        cfg_path = run_dir / "config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        print(f"\n=== {name} (Durchlauf {rep + 1}/{a.repeats}) ===", flush=True)
        start = time.time()
        log = run_dir / "train.log"
        with log.open("w", encoding="utf-8") as f:
            rc = subprocess.run([str(a.exe), str(cfg_path),
                                 "--timestep-limit", str(seed_steps + a.steps)],
                                cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        wall = time.time() - start

        metrics_path = run_dir / "metrics.csv"
        if rc != 0 or not metrics_path.exists():
            print(f"  FEHLER (exit {rc}), siehe {log}")
            continue
        data = read_rows(metrics_path)[a.warmup:]
        if not data:
            print("  zu wenige Iterationen")
            continue

        def mean(key: str) -> float:
            vals = [float(r[key]) for r in data if r.get(key)]
            return sum(vals) / len(vals) if vals else float("nan")

        per_iteration = mean("Timesteps Collected")
        row = {
            "variante": name,
            "durchlauf": rep + 1,
            "iterationen": len(data),
            "steps_pro_iteration": round(per_iteration),
            "iter_s": round(mean("Total Iteration Time"), 3),
            "infer_s": round(mean("Policy Infer Time"), 3),
            "sim_s": round(mean("Env Step Time"), 3),
            "lernen_s": round(mean("Consumption Time"), 3),
            "sps_report": round(per_iteration / mean("Total Iteration Time")),
            "sps_wall": round(a.steps / wall),
            "wall_s": round(wall, 1),
        }
        rows.append(row)
        print(f"  Iteration {row['iter_s']}s (Inferenz {row['infer_s']}, Sim {row['sim_s']}, "
              f"Lernen {row['lernen_s']}) -> {row['sps_report']:,} SPS", flush=True)
        shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)

    if not rows:
        return
    out = RESULTS / f"config_tuning{a.tag}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Über die Wiederholungen mitteln und die Streuung ausweisen
    summary = {}
    for row in rows:
        summary.setdefault(row["variante"], []).append(row["sps_report"])

    means = {name: sum(vals) / len(vals) for name, vals in summary.items()}
    reference = means.get(a.variants[0], next(iter(means.values())))

    print(f"\n{'Variante':<22}{'SPS (Mittel)':>14}{'Streuung':>11}{'Läufe':>7}{'gegen erste':>13}")
    for name in sorted(means, key=lambda n: -means[n]):
        vals = summary[name]
        spread = (max(vals) - min(vals)) / means[name] if len(vals) > 1 else 0.0
        print(f"{name:<22}{means[name]:>14,.0f}{spread:>10.1%}{len(vals):>7}"
              f"{means[name] / reference - 1:>12.1%}")

    if any(len(v) > 1 for v in summary.values()):
        worst = max((max(v) - min(v)) / (sum(v) / len(v)) for v in summary.values() if len(v) > 1)
        print(f"\nHinweis: Streuung identischer Läufe bis {worst:.1%} - Unterschiede darunter "
              f"sind nicht belastbar.")
    print(f"Geschrieben: {out}")


if __name__ == "__main__":
    main()
