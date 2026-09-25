"""Schreibt die Zusammenfassung eines Experiment-Laufs als summary.json + summary.md.

    python tools/experiments/summarize.py --run runs/exp_x --out results/exp_x
        [--name x] [--config train/configs/experiments/x.json] [--start-checkpoint <pfad>]
        [--duel-start duel_start.json] [--duel-baseline duel_baseline.json] [--wall-seconds N]
        [--abort-reason "..."]

Alles, was compare.py später braucht, steht in summary.json; summary.md ist für Menschen.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_util import KEY_COLUMNS, column, count_bad, mean, read_rows, window_mean  # noqa: E402


def running_stats(checkpoint: Path | None) -> dict:
    """Return-std und Zähler aus RUNNING_STATS.json (Audit: Return-std 15,12 lokal nachmessen)."""
    if not checkpoint:
        return {}
    path = checkpoint / "RUNNING_STATS.json"
    if not path.exists():
        return {}
    j = json.loads(path.read_text(encoding="utf-8"))
    rrs = j.get("reward_running_stats", {})
    var = rrs.get("var", [None])[0]
    count = rrs.get("count")
    std = math.sqrt(var / (count - 1)) if var is not None and count and count > 1 else None
    return {"return_std": std, "count": count, "cumulative_timesteps": j.get("cumulative_timesteps"),
            "cumulative_model_updates": j.get("cumulative_model_updates"),
            "skill_rating": j.get("skill_rating")}


def summarize_metrics(rows: list[dict[str, str]]) -> dict:
    out: dict = {"iterations": len(rows)}
    if not rows:
        return out
    out["steps_first"] = column(rows[:1], "Cumulative Timesteps")[0]
    out["steps_last"] = column(rows[-1:], "Cumulative Timesteps")[0]
    out["steps_run"] = out["steps_last"] - out["steps_first"] + column(rows[:1], "Timesteps Collected")[0]
    last, first = {}, {}
    for key, short in KEY_COLUMNS:
        last[short] = window_mean(rows, key)
        first[short] = mean(column(rows[:max(20, len(rows) // 5)], key))
    out["last_20pct"] = {k: (None if math.isnan(v) else v) for k, v in last.items()}
    out["first_20pct"] = {k: (None if math.isnan(v) else v) for k, v in first.items()}
    # Kein Abbruchgrund (check_abort.py), aber sichtbar: Iterationen ohne beendete Episode
    out["ep_reward_nan_iterations"] = count_bad(rows, "Average Episode Reward")
    return out


def load_duel(path: Path | None) -> dict | None:
    if not path or not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    total = d.get("goals_a", 0) + d.get("goals_b", 0)
    share = d["goals_a"] / total if total else 0.5
    se = math.sqrt(share * (1 - share) / total) if total else None
    d["goal_share_a"] = share
    d["goal_share_se"] = se
    return d


def fmt(v, digits=4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:,.{digits}f}" if abs(v) < 1e5 else f"{v:,.0f}"
    return str(v)


def to_markdown(s: dict) -> str:
    lines = [f"# Experiment {s['name']}", "",
             f"Config: `{s.get('config')}`  ", f"Start-Checkpoint: `{s.get('start_checkpoint')}`  ",
             f"Lauf-Ordner: `{s.get('run_dir')}`  ", f"Erzeugt: {s['created']}", ""]
    if s.get("abort_reason"):
        lines += [f"**ABGEBROCHEN:** {s['abort_reason']}", ""]
    m = s.get("metrics", {})
    lines += ["## Verlauf", "",
              f"Iterationen: {m.get('iterations')}, Steps im Lauf: {fmt(m.get('steps_run'), 0)}, "
              f"Wall-Clock: {fmt(s.get('wall_seconds'), 0)} s, "
              f"Iterationen ohne beendete Episode (Average Episode Reward nan/leer): "
              f"{m.get('ep_reward_nan_iterations', '-')}", ""]
    if m.get("last_20pct"):
        lines += ["| Größe | erstes Fünftel | letztes Fünftel |", "|---|---|---|"]
        for _, short in KEY_COLUMNS:
            lines.append(f"| {short} | {fmt(m['first_20pct'].get(short))} | {fmt(m['last_20pct'].get(short))} |")
        lines.append("")
    rs = s.get("running_stats_start") or {}
    if rs:
        lines += ["## RUNNING_STATS.json des Start-Checkpoints", "",
                  f"Return-std {fmt(rs.get('return_std'))}, Zähler {rs.get('count')}, "
                  f"Steps {rs.get('cumulative_timesteps')}, Updates {rs.get('cumulative_model_updates')}", ""]
    for label, key in (("Duell Ende gegen Start", "duel_start"), ("Duell Ende gegen Baseline-Ende", "duel_baseline")):
        d = s.get(key)
        if d:
            lines += [f"## {label}", "",
                      f"Tore {d['goals_a']}:{d['goals_b']} in {d['games']} Spielen, Toranteil A "
                      f"{d['goal_share_a']:.1%} ± {fmt(d['goal_share_se'], 3)} (SE), "
                      f"Siege {d['wins_a']}:{d['wins_b']} ({d['draws']} remis)", ""]
    if s.get("ratings"):
        lines += ["## TrueSkill (Ladder im Lauf-Ordner)", "", "| Checkpoint | mu | sigma | mu-3sigma |", "|---|---|---|---|"]
        for name, r in s["ratings"].items():
            lines.append(f"| {name} | {r['mu']:.2f} | {r['sigma']:.2f} | {r['conservative']:.2f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--start-checkpoint", type=Path, default=None)
    ap.add_argument("--end-checkpoint", type=Path, default=None)
    ap.add_argument("--duel-start", type=Path, default=None)
    ap.add_argument("--duel-baseline", type=Path, default=None)
    ap.add_argument("--wall-seconds", type=float, default=None)
    ap.add_argument("--abort-reason", default=None)
    a = ap.parse_args()

    metrics_path = a.run / "metrics.csv"
    rows = read_rows(metrics_path) if metrics_path.exists() else []
    ratings_path = a.run / "ratings.json"
    summary = {
        "name": a.name or a.run.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(a.run),
        "config": str(a.config) if a.config else None,
        "start_checkpoint": str(a.start_checkpoint) if a.start_checkpoint else None,
        "end_checkpoint": str(a.end_checkpoint) if a.end_checkpoint else None,
        "wall_seconds": a.wall_seconds,
        "abort_reason": a.abort_reason,
        "metrics": summarize_metrics(rows),
        "running_stats_start": running_stats(a.start_checkpoint),
        "duel_start": load_duel(a.duel_start),
        "duel_baseline": load_duel(a.duel_baseline),
        "ratings": json.loads(ratings_path.read_text(encoding="utf-8")) if ratings_path.exists() else None,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (a.out / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
