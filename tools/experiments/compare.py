"""Vergleicht Experimente mit der Baseline und gibt eine Markdown-Tabelle aus.

    python tools/experiments/compare.py results/exp_baseline_2026-10-01 results/exp_h2_ent_coef_0004_2026-10-01 ...
    python tools/experiments/compare.py results/exp_* --baseline results/exp_baseline_2026-10-01 --out results/compare.md

Jeder Ordner braucht summary.json (aus summarize.py) und, wenn vorhanden, metrics.csv.
Verglichen werden (letztes Fünftel der Iterationen): ep_end_goal, ep_end_timeout, Entropie,
Clip-Fraction, KL, Value Loss, Truncated Steps, SPS - dazu Ladder/TrueSkill (mu - 3 sigma des
End-Checkpoints, Unsicherheit = sigma) und das direkte Duell gegen das Baseline-Ende mit
Standardfehler des Toranteils. Die Baseline ist der Ordner mit --baseline oder der, dessen
Name mit "exp_baseline" beginnt.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_util import read_rows, window_mean  # noqa: E402

COMPARE_COLUMNS = [
    # (Kurzname in summary.json, Anzeigename, Nachkommastellen, "hoch ist gut" oder None)
    ("ep_end_goal", "Tor-Anteil", 3, True),
    ("ep_end_timeout", "Timeout-Anteil", 3, False),
    ("ep_length_steps", "Ep.-Länge", 0, None),
    ("ball_touch", "Ballkontakt", 4, True),
    ("entropy", "Entropie", 3, None),
    ("clip_fraction", "Clip-Frac.", 4, None),
    ("kl", "KL", 5, None),
    ("value_loss", "Value Loss", 3, None),
    ("val_target", "Val Target", 2, None),
    ("truncated_steps", "Trunc. Steps", 0, None),
    ("sps", "SPS", 0, None),
]


def load_experiment(folder: Path) -> dict:
    summary_path = folder / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"summary.json fehlt in {folder} (tools/experiments/summarize.py)")
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    s["_folder"] = folder
    # Wenn metrics.csv daneben liegt, das letzte Fünftel frisch berechnen (robuster als summary)
    metrics = folder / "metrics.csv"
    if metrics.exists():
        rows = read_rows(metrics)
        if rows:
            from metrics_util import KEY_COLUMNS
            s.setdefault("metrics", {})["last_20pct"] = {
                short: (None if math.isnan(v) else v)
                for key, short in KEY_COLUMNS for v in [window_mean(rows, key)]}
            s["metrics"]["iterations"] = len(rows)
    return s


def end_rating(s: dict) -> tuple[float, float] | None:
    """(mu - 3 sigma, sigma) des End-Checkpoints, wenn eine Ladder im Lauf-Ordner lief."""
    ratings = s.get("ratings") or {}
    if not ratings:
        return None
    # End-Checkpoint = höchste Step-Zahl im Schlüssel "<lauf>/<steps>"
    def steps(name: str) -> int:
        try:
            return int(name.rsplit("/", 1)[-1])
        except ValueError:
            return -1
    name = max(ratings, key=steps)
    r = ratings[name]
    return r["conservative"], r["sigma"]


def fmt(v, digits) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:,.{digits}f}"


def delta(v, base, digits) -> str:
    if v is None or base is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(base, float) and math.isnan(base)):
        return "-"
    d = v - base
    sign = "+" if d >= 0 else ""
    if base not in (0, 0.0):
        return f"{sign}{d:,.{digits}f} ({sign}{d / abs(base):.1%})"
    return f"{sign}{d:,.{digits}f}"


def build_table(experiments: list[dict], baseline: dict) -> str:
    lines = []
    header = "| Experiment | Iter. | " + " | ".join(name for _, name, _, _ in COMPARE_COLUMNS) + " | TrueSkill (mu-3σ ± σ) | Duell gg. Baseline-Ende (Toranteil ± SE) |"
    lines.append(header)
    lines.append("|" + "---|" * (header.count("|") - 1))
    base_last = baseline.get("metrics", {}).get("last_20pct", {})
    for s in experiments:
        last = s.get("metrics", {}).get("last_20pct", {})
        is_base = s is baseline
        cells = [s["name"] + (" (Baseline)" if is_base else ""), str(s.get("metrics", {}).get("iterations", "-"))]
        for short, _, digits, _ in COMPARE_COLUMNS:
            v = last.get(short)
            if is_base:
                cells.append(fmt(v, digits))
            else:
                cells.append(f"{fmt(v, digits)} ({delta(v, base_last.get(short), digits)})" if v is not None else "-")
        r = end_rating(s)
        cells.append(f"{r[0]:.2f} ± {r[1]:.2f}" if r else "-")
        d = s.get("duel_baseline")
        if d and not is_base:
            se = d.get("goal_share_se")
            cells.append(f"{d['goal_share_a']:.1%} ± {(se or 0) * 100:.1f} pp ({d['goals_a']}:{d['goals_b']}, {d['games']} Spiele)")
        else:
            cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def verdict_hints(experiments: list[dict], baseline: dict) -> str:
    """Kurze, regelbasierte Hinweise; die Entscheidung bleibt beim Menschen (AUDIT.md Roadmap)."""
    out = ["", "## Hinweise (regelbasiert, keine Entscheidung)", ""]
    base_last = baseline.get("metrics", {}).get("last_20pct", {})
    for s in experiments:
        if s is baseline:
            continue
        notes = []
        last = s.get("metrics", {}).get("last_20pct", {})
        if s.get("abort_reason"):
            notes.append(f"abgebrochen: {s['abort_reason']}")
        d = s.get("duel_baseline")
        if d and d.get("goal_share_se") is not None:
            share, se = d["goal_share_a"], d["goal_share_se"]
            if share - 2 * se > 0.5:
                notes.append(f"Duell: signifikant besser als Baseline ({share:.1%}, 2 SE = {2 * se:.1%})")
            elif share + 2 * se < 0.5:
                notes.append(f"Duell: signifikant schlechter als Baseline ({share:.1%})")
            else:
                notes.append(f"Duell: kein signifikanter Unterschied ({share:.1%} ± {2 * se:.1%}); mehr Spiele oder längerer Lauf")
        g, gb = last.get("ep_end_goal"), base_last.get("ep_end_goal")
        if g is not None and gb is not None and gb > 0:
            notes.append(f"Tor-Anteil {g:.3f} gegen {gb:.3f} ({(g - gb) / gb:+.1%})")
        e = last.get("entropy")
        if e is not None and e < 2.5:
            notes.append(f"Entropie {e:.2f} < 2,5: Kollaps-Schwelle aus AUDIT.md H2")
        sps, spsb = last.get("sps"), base_last.get("sps")
        if sps and spsb and abs(sps / spsb - 1) > 0.07:
            notes.append(f"SPS {sps / spsb - 1:+.1%} gegenüber Baseline (über der 7-%-Streuung)")
        out.append(f"* **{s['name']}**: " + ("; ".join(notes) if notes else "keine Auffälligkeiten"))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="Markdown zusätzlich in Datei schreiben")
    a = ap.parse_args()

    experiments = [load_experiment(f) for f in a.folders]
    if a.baseline:
        baseline = load_experiment(a.baseline)
        if not any(e["_folder"] == baseline["_folder"] for e in experiments):
            experiments.insert(0, baseline)
        else:
            baseline = next(e for e in experiments if e["_folder"] == baseline["_folder"])
    else:
        candidates = [e for e in experiments if e["_folder"].name.startswith("exp_baseline") or e["name"] == "baseline"]
        if not candidates:
            raise SystemExit("Keine Baseline gefunden: --baseline angeben oder Ordner exp_baseline_* mitgeben")
        baseline = candidates[0]

    md = "# Experiment-Vergleich (letztes Fünftel der Iterationen)\n\n"
    md += f"Baseline: `{baseline['_folder']}`\n\n"
    md += build_table(experiments, baseline) + "\n"
    md += verdict_hints(experiments, baseline) + "\n"
    print(md)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
