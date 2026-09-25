"""Bestandsaufnahme eines Laufs für die Übergabe (LOCAL_RUNBOOK Schritt 6):

    python tools/local/inspect_run.py --run runs/lucy_1v1 --out results/local_check_<datum>/run_inspect.json

Liest ohne Training zu starten: Checkpoints (Step-Zahl, Größe), RUNNING_STATS.json des neuesten
Checkpoints (Return-std, Zähler, Modell-Updates, Skill-Rating), metrics.csv (Iterationen,
Kopfzeilen, letztes Fünftel der Kernmetriken). Das sind genau die Werte, die AUDIT.md §0 als
"lokal nachmessen" markiert (2,70 Mrd. Steps, Return-std 15,12, 5 Kopfzeilen, ...).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "experiments"))
from metrics_util import KEY_COLUMNS, read_rows, window_mean  # noqa: E402


def inspect(run_dir: Path) -> dict:
    out: dict = {"run": str(run_dir)}
    folder = run_dir / "checkpoints"
    ckpts = sorted((d for d in folder.glob("*") if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name)) \
        if folder.exists() else []
    out["checkpoints"] = [{"steps": int(d.name), "mb": round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6, 1),
                           "has_policy": (d / "PPO_POLICY.lt").exists(),
                           "has_running_stats": (d / "RUNNING_STATS.json").exists()} for d in ckpts]
    if ckpts:
        latest = ckpts[-1]
        out["latest_checkpoint"] = str(latest)
        rs_path = latest / "RUNNING_STATS.json"
        if rs_path.exists():
            j = json.loads(rs_path.read_text(encoding="utf-8"))
            rrs = j.get("reward_running_stats", {})
            var = rrs.get("var", [None])[0]
            count = rrs.get("count")
            out["running_stats"] = {
                "cumulative_timesteps": j.get("cumulative_timesteps"),
                "cumulative_model_updates": j.get("cumulative_model_updates"),
                "epoch": j.get("epoch"),
                "skill_rating": j.get("skill_rating"),
                "return_mean": rrs.get("mean", [None])[0],
                "return_std": math.sqrt(var / (count - 1)) if var is not None and count and count > 1 else None,
                "count": count,
            }
    metrics = run_dir / "metrics.csv"
    if metrics.exists():
        text = metrics.read_text(encoding="utf-8", errors="replace")
        # Kopfzeilen beginnen mit einem Anführungszeichen, Datenzeilen mit einer Zahl (oder leer)
        header_lines = sum(1 for line in text.splitlines() if line.startswith('"'))
        rows = read_rows(metrics)
        info = {"iterations": len(rows), "header_lines": header_lines,
                "first_steps": rows[0].get("Cumulative Timesteps") if rows else None,
                "last_steps": rows[-1].get("Cumulative Timesteps") if rows else None,
                "columns": list(rows[0].keys()) if rows else []}
        if rows:
            info["last_20pct"] = {short: (None if math.isnan(v) else v)
                                  for key, short in KEY_COLUMNS for v in [window_mean(rows, key)]}
            nan_rows = sum(1 for r in rows if any(v == "nan" for v in r.values()))
            info["rows_with_nan"] = nan_rows
        out["metrics"] = info
    used = run_dir / "config_used.json"
    if used.exists():
        cfg = json.loads(used.read_text(encoding="utf-8"))
        out["config_used"] = {"_git": cfg.get("_git"), "_started": cfg.get("_started"),
                              "game_timeout_secs": cfg.get("env", {}).get("game_timeout_secs"),
                              "ent_coef": cfg.get("learner", {}).get("ent_coef"),
                              "exp_buffer_iterations": cfg.get("learner", {}).get("exp_buffer_iterations")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "runs" / "lucy_1v1")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if not a.run.exists():
        print(f"Lauf-Ordner fehlt: {a.run}")
        return 1
    info = inspect(a.run)
    text = json.dumps(info, indent=2, ensure_ascii=False)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
