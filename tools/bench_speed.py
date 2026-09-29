"""SPS-Vergleich von Trainer-Varianten mit Wiederholungen (Geschwindigkeit, AUDIT.md §9).

    python tools/bench_speed.py --plan plan.json --tag g1 [--steps 15000000] [--repeats 3]

Jede Variante startet von einer Kopie desselben Checkpoints (der Quellordner wird nur gelesen),
trainiert `--steps` Steps und wird über die Iterationen nach `--warmup` ausgewertet. Die Varianten
laufen abwechselnd (A B C A B C ...), damit Takt- und Temperaturdrift alle gleich trifft.

Plan (JSON), je Variante optional: "exe" (Trainer, Default build/cpp_cu128/train_bot.exe),
"config" (Überschreibungen der Basis-Config, verschachtelt), "env" (Umgebungsvariablen):

    {"base_config": "train/configs/lucy_1v1_zero_sum_drill.json",
     "variants": {"alt": {"exe": "build/cpp_cu128/train_bot.exe"},
                  "overlap": {"exe": "build/cpp_cu128_speed/train_bot.exe",
                              "config": {"learner": {"collection_during_learn": true}}}}}

Ergebnis: results/speed_<tag>/ mit metrics je Lauf, runs.csv (ein Lauf je Zeile) und summary.md
(Mittel und Standardabweichung je Variante). Läufe liegen in runs/speed_<tag>_<variante>_<n>/;
es wird nichts gelöscht, ein vorhandener Ordner ist ein Fehler.

SPS = Steps pro Iteration / "Total Iteration Time" (Wanduhrzeit von Iteration zu Iteration, gilt
auch mit collection_during_learn). Zusätzlich die Wanduhr-SPS des ganzen Prozesses (mit Start).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from show_metrics import read_rows  # noqa: E402

DEFAULT_EXE = "build/cpp_cu128/train_bot.exe"

# Zeitspalten, die in der Zusammenfassung erscheinen (fehlende werden übersprungen)
TIME_COLUMNS = [
    "Total Iteration Time", "Collection Time", "Consumption Time", "Collect-Consume Overlap Time",
    "Policy Infer Time", "Infer Call Time", "Traj Append Time", "Obs Tensor Time", "Obs To Device Time",
    "Env Step Time", "Step Callback Time", "Play Stats Time", "Agent Wait Time",
    "Collect Concat Time", "Add Experience Time", "Exp Value Pred Time", "Exp GAE Time", "Exp Submit Time",
    "PPO Learn Time", "PPO Shuffle Time", "PPO Minibatch Time", "PPO Optim Time", "PPO Param Copy Time",
    "Empty Cache Time", "Prev Tail Time", "Prev Skill Eval Time", "Prev Iteration Callback Time", "Prev Save Time",
    "Infer Policy Sync Time",
    # keine Zeiten: Iterationsgröße und wie viel davon während der Lernphase gesammelt wurde (G5)
    "Timesteps Collected", "Steps Collected During Learn",
]


def deep_merge(base: dict, patch: dict) -> dict:
    out = json.loads(json.dumps(base))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def latest_checkpoint(folder: Path) -> Path:
    dirs = [d for d in folder.iterdir() if d.is_dir() and d.name.isdigit() and (d / "PPO_POLICY.lt").exists()]
    if not dirs:
        raise SystemExit(f"Keine Checkpoints in {folder}")
    return max(dirs, key=lambda d: int(d.name))


def mean_of(rows: list[dict], key: str) -> float | None:
    vals = []
    for r in rows:
        v = r.get(key)
        if v not in (None, "", "nan"):
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else None


def evaluate(metrics: Path, warmup: int) -> dict:
    rows = read_rows(metrics)[warmup:]
    if not rows:
        return {}
    out = {"iterations": len(rows)}
    steps = mean_of(rows, "Timesteps Collected")
    it = mean_of(rows, "Total Iteration Time")
    out["sps"] = steps / it if steps and it else None
    for c in TIME_COLUMNS:
        m = mean_of(rows, c)
        if m is not None:
            out[c] = m
    for c in ("Policy Entropy", "Mean KL Divergence", "SB3 Clip Fraction", "Value Function Loss"):
        m = mean_of(rows, c)
        if m is not None:
            out[c] = m
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=15_000_000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10, help="verworfene Iterationen am Anfang")
    ap.add_argument("--start", type=Path, default=None,
                    help="Start-Checkpoint (Default: neuester in runs/lucy_1v1/checkpoints, nur gelesen)")
    ap.add_argument("--variants", nargs="*", default=None, help="nur diese Varianten des Plans")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    plan = json.loads(a.plan.read_text(encoding="utf-8"))
    base = json.loads((ROOT / plan.get("base_config", "train/configs/lucy_1v1_zero_sum_drill.json"))
                      .read_text(encoding="utf-8"))
    variants = plan["variants"]
    names = a.variants or list(variants)
    start = a.start or latest_checkpoint(ROOT / "runs" / "lucy_1v1" / "checkpoints")
    out_dir = ROOT / "results" / f"speed_{a.tag}"
    order = [(rep, name) for rep in range(1, a.repeats + 1) for name in names]
    for rep, name in order:
        run = ROOT / "runs" / f"speed_{a.tag}_{name}_{rep}"
        if run.exists():
            raise SystemExit(f"existiert schon: {run}")
    print(f"Start-Checkpoint {start} (nur gelesen), {a.steps:,} Steps je Lauf, "
          f"{len(order)} Läufe: {' '.join(f'{n}#{r}' for r, n in order)}")
    if a.dry_run:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")

    py_home = subprocess.check_output([sys.executable, "-c", "import sys; print(sys.base_prefix)"], text=True).strip()
    results = []
    for rep, name in order:
        v = variants[name]
        run = ROOT / "runs" / f"speed_{a.tag}_{name}_{rep}"
        shutil.copytree(start, run / "checkpoints" / start.name)
        cfg = deep_merge(base, v.get("config", {}))
        cfg["learner"]["checkpoint_folder"] = f"runs/speed_{a.tag}_{name}_{rep}/checkpoints"
        cfg["learner"]["extra_steps"] = a.steps
        cfg["learner"]["timestep_limit"] = 0
        cfg["learner"]["timesteps_per_save"] = 10 ** 12
        cfg["learner"]["save_on_exit"] = False
        cfg["metrics"]["run"] = f"speed_{a.tag}_{name}"
        cfg_path = run / "config.json"
        cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        env = dict(os.environ, PYTHONHOME=py_home, PATH=f"{py_home};{os.environ['PATH']}")
        env.update({k: str(x) for k, x in v.get("env", {}).items()})
        exe = ROOT / v.get("exe", DEFAULT_EXE)
        print(f"{time.strftime('%H:%M:%S')} {name} #{rep} ({exe.relative_to(ROOT)})", flush=True)
        t0 = time.time()
        with (run / "train.log").open("w", encoding="utf-8") as log:
            rc = subprocess.run([str(exe), str(cfg_path)], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT).returncode
        wall = time.time() - t0
        metrics = run / "metrics.csv"
        if rc != 0 or not metrics.exists():
            print(f"  FEHLER exit {rc}, siehe {run / 'train.log'}")
            results.append({"variant": name, "rep": rep, "exit": rc})
            continue
        shutil.copyfile(metrics, out_dir / f"{name}_{rep}.csv")
        ev = evaluate(metrics, a.warmup)
        ev.update({"variant": name, "rep": rep, "exit": rc, "wall_s": round(wall, 1),
                   "sps_wall": a.steps / wall})
        results.append(ev)
        print(f"  {ev.get('sps', 0):,.0f} SPS (Iteration {ev.get('Total Iteration Time', 0):.3f} s, "
              f"Sammeln {ev.get('Collection Time', 0):.3f}, Lernen {ev.get('Consumption Time', 0):.3f}), "
              f"Wanduhr {ev['sps_wall']:,.0f}", flush=True)

    keys = ["variant", "rep", "exit", "iterations", "sps", "sps_wall", "wall_s"] + \
        [k for k in TIME_COLUMNS + ["Policy Entropy", "Mean KL Divergence", "SB3 Clip Fraction", "Value Function Loss"]
         if any(k in r for r in results)]
    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    lines = [f"# SPS-Vergleich `{a.tag}`", "",
             f"Start {start.name}, {a.steps:,} Steps je Lauf, {a.repeats} Wiederholungen abwechselnd, "
             f"erste {a.warmup} Iterationen verworfen. SPS = Steps/Iteration ÷ Total Iteration Time.", "",
             "| Variante | SPS Mittel | SD | Läufe | gegen erste | Iteration s | Sammeln s | Lernen s |",
             "|---|---|---|---|---|---|---|---|"]
    ok = [r for r in results if r.get("exit") == 0 and r.get("sps")]
    by = {n: [r for r in ok if r["variant"] == n] for n in names}
    ref = st.mean(r["sps"] for r in by[names[0]]) if by[names[0]] else None
    for n in names:
        rs = by[n]
        if not rs:
            lines.append(f"| {n} | FEHLER | | 0 | | | | |")
            continue
        m = st.mean(r["sps"] for r in rs)
        sd = st.stdev(r["sps"] for r in rs) if len(rs) > 1 else 0.0
        rel = f"{m / ref - 1:+.1%}" if ref else ""
        lines.append(f"| {n} | {m:,.0f} | {sd:,.0f} | {len(rs)} | {rel} | "
                     f"{st.mean(r['Total Iteration Time'] for r in rs):.3f} | "
                     f"{st.mean(r['Collection Time'] for r in rs):.3f} | {st.mean(r['Consumption Time'] for r in rs):.3f} |")
    lines += ["", "Zeitaufschlüsselung (Mittel über die Läufe, Sekunden je Iteration; Agent-Zeiten je Thread):", "",
              "| Spalte | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for c in TIME_COLUMNS:
        if not any(c in r for r in ok):
            continue
        cells = []
        for n in names:
            vals = [r[c] for r in by[n] if c in r]
            cells.append(f"{st.mean(vals):.3f}" if vals else "")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
