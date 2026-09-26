"""Vergleicht Experimente mit der Baseline und gibt eine Markdown-Tabelle aus.

    python tools/experiments/compare.py results/exp_baseline_2026-10-01 results/exp_h2_ent_coef_0004_2026-10-01 ...
    python tools/experiments/compare.py results/exp_* --baseline results/exp_baseline_2026-10-01 --out results/compare.md

Jeder Ordner braucht summary.json (aus summarize.py) und, wenn vorhanden, metrics.csv.

**Hauptkriterium** ist das direkte Duell jedes Experiment-Endes gegen das Baseline-Ende
(duel_end_vs_baseline.json aus run_experiment.ps1): mittlere **Tordifferenz pro Spiel** (Spiele à
300 s, siehe eval/cpp/duel.cpp) mit 95-%-t-Intervall über die Einzelspiele. Liegt das ganze
Intervall über 0, ist das Experiment besser; ganz darunter schlechter; sonst unklar (im Rauschen).
Zusätzlich: Gewinnrate (Remis = halber Sieg, Wilson-Intervall) und Tore pro Minute. Die Gewinnrate
war vorher das Hauptkriterium (R12), taugte aber nicht: Mit 120-s-Spielen, die beim ersten Tor
endeten, waren fast alle Spiele remis.

**TrueSkill** kommt nur aus EINER gemeinsamen Ladder, die compare.py selbst spielt: alle
Experiment-Enden plus Baseline-Start und Baseline-Ende, jeder gegen jeden (--ladder-games Spiele je
Paarung, 0 = keine Ladder). Die Ladders in den einzelnen Lauf-Ordnern (ratings.json) haben jeweils
eigene Nullpunkte und sind untereinander nicht vergleichbar; compare.py benutzt sie nicht.

Dazu je Experiment (letztes Fünftel der Iterationen): ep_end_goal, ep_end_timeout, Entropie,
Clip-Fraction, KL, Value Loss, Truncated Steps, SPS mit Differenz zur Baseline. Experimente, die mehr
als einen Config-Wert ändern, werden als Bündel markiert (Review R11). Die Baseline ist der Ordner
mit --baseline oder der, dessen Name mit "exp_baseline" beginnt.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_util import duel_stats, read_rows, window_mean  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

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

# Felder, die run_experiment.ps1 pro Lauf setzt, und Anmerkungen: kein Unterschied zwischen
# Experimenten, zählen also nicht als Änderung gegenüber der Baseline.
RUN_KEYS = {"learner.checkpoint_folder", "learner.extra_steps", "learner.save_on_exit",
            "learner.timestep_limit", "metrics.run"}

BASELINE_START, BASELINE_END = "Baseline-Start", "Baseline-Ende"


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def load_config(s: dict) -> dict | None:
    """Config des Laufs: config.json im Ergebnisordner (von run_experiment.ps1 abgeleitet), sonst
    die Original-Config aus summary.json."""
    candidates = [s["_folder"] / "config.json"]
    if s.get("config"):
        candidates.append(Path(s["config"]))
    for path in candidates:
        if path.exists():
            return flatten(json.loads(path.read_text(encoding="utf-8-sig")))   # PowerShell schreibt ein BOM
    return None


def config_changes(s: dict, baseline: dict) -> dict[str, tuple] | None:
    """Alle Config-Werte, in denen sich ein Experiment von der Baseline unterscheidet."""
    a, b = load_config(baseline), load_config(s)
    if a is None or b is None:
        return None
    keys = {k for k in set(a) | set(b) if k not in RUN_KEYS and not k.rsplit(".", 1)[-1].startswith("_")}
    return {k: (a.get(k), b.get(k)) for k in sorted(keys) if a.get(k) != b.get(k)}


def bundle_label(changes: dict | None) -> str:
    """Review-Befund R11: Ein Experiment mit mehr als einer Änderung ist ein Bündel. Sein Ergebnis
    lässt sich keinem einzelnen Wert zuordnen (K3: 7 Reward-Gewichte)."""
    if changes and len(changes) > 1:
        return f" **(Bündel: {len(changes)} Werte)**"
    return ""


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


def duel_summary(d: dict | None) -> dict | None:
    """Kennzahlen des Duells aus Sicht von A (= Experiment-Ende), siehe metrics_util.duel_stats."""
    if not d or not d.get("games"):
        return None
    return duel_stats(d)


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


# --- gemeinsame Ladder -------------------------------------------------------------------

def end_label(s: dict, baseline: dict) -> str:
    return BASELINE_END if s is baseline else f"{s['name']}-Ende"


def ladder_participants(experiments: list[dict], baseline: dict) -> dict[str, Path]:
    """Label -> PPO_POLICY.lt: Baseline-Start, Baseline-Ende und jedes Experiment-Ende."""
    out: dict[str, Path] = {}

    def add(label: str, ckpt: str | None):
        if not ckpt:
            return
        policy = Path(ckpt) / "PPO_POLICY.lt"
        if policy.exists():
            out[label] = policy

    add(BASELINE_START, baseline.get("start_checkpoint"))
    for s in experiments:
        label = end_label(s, baseline)
        if label in out:
            label = f"{label} ({s['_folder'].name})"
        add(label, s.get("end_checkpoint"))
    return out


def run_joint_ladder(participants: dict[str, Path], games: int, exe: Path, out_path: Path | None) -> dict:
    """Jeder gegen jeden, ein gemeinsames TrueSkill-Modell. Liefert {"ratings": {...}, "duels": [...]}."""
    sys.path.insert(0, str(ROOT))
    from eval.ladder import ENV, conservative, run_duel, update_ratings  # noqa: E402

    ratings = {label: ENV.create_rating() for label in participants}
    duels = []
    for (la, pa), (lb, pb) in itertools.combinations(participants.items(), 2):
        print(f"Ladder: {la} gegen {lb} ({games} Spiele) ...", file=sys.stderr)
        result = run_duel(pa, pb, games, exe=exe)
        result.name_a, result.name_b = la, lb
        ratings = update_ratings(ratings, result)
        duels.append({"a": la, "b": lb, "games": result.games, "goals_a": result.goals_a,
                      "goals_b": result.goals_b, "wins_a": result.wins_a, "wins_b": result.wins_b,
                      "draws": result.draws})
    data = {"games_per_pair": games,
            "participants": {label: str(path) for label, path in participants.items()},
            "ratings": {label: {"mu": r.mu, "sigma": r.sigma, "conservative": conservative(r)}
                        for label, r in ratings.items()},
            "duels": duels}
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


# --- Ausgabe -------------------------------------------------------------------------------

def build_table(experiments: list[dict], baseline: dict, ladder: dict | None) -> str:
    lines = []
    header = ("| Experiment | Iter. | **Duell gg. Baseline-Ende: Tordifferenz/Spiel [95-%-KI]** | "
              "Gewinnrate [95-%-KI] | Tore/min (Exp. : Base) | "
              + " | ".join(name for _, name, _, _ in COMPARE_COLUMNS)
              + " | TrueSkill gemeinsame Ladder (mu-3σ ± σ) |")
    lines.append(header)
    lines.append("|" + "---|" * (header.count("|") - 1))
    base_last = baseline.get("metrics", {}).get("last_20pct", {})
    ratings = (ladder or {}).get("ratings", {})
    for s in experiments:
        last = s.get("metrics", {}).get("last_20pct", {})
        is_base = s is baseline
        label = s["name"] + (" (Baseline)" if is_base else bundle_label(config_changes(s, baseline)))
        cells = [label, str(s.get("metrics", {}).get("iterations", "-"))]
        ds = None if is_base else duel_summary(s.get("duel_baseline"))
        if ds:
            d = s["duel_baseline"]
            cells.append(f"**{ds['goal_diff']:+.3f}** [{ds['goal_diff_ci_low']:+.3f}, {ds['goal_diff_ci_high']:+.3f}] "
                         f"({ds['games']} Spiele)")
            cells.append(f"{ds['win_rate']:.1%} [{ds['win_rate_ci_low']:.1%}, {ds['win_rate_ci_high']:.1%}] "
                         f"({d['wins_a']}:{d['wins_b']}, {d['draws']} remis)")
            cells.append(f"{ds['goals_per_minute_a']:.3f} : {ds['goals_per_minute_b']:.3f}")
        else:
            cells += ["(Referenz)" if is_base else "-", "-", "-"]
        for short, _, digits, _ in COMPARE_COLUMNS:
            v = last.get(short)
            if is_base:
                cells.append(fmt(v, digits))
            else:
                cells.append(f"{fmt(v, digits)} ({delta(v, base_last.get(short), digits)})" if v is not None else "-")
        r = ratings.get(end_label(s, baseline))
        cells.append(f"{r['conservative']:.2f} ± {r['sigma']:.2f}" if r else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ladder_section(ladder: dict | None, reason: str | None) -> str:
    out = ["", "## Gemeinsame TrueSkill-Ladder", ""]
    if not ladder:
        out.append(f"Nicht gespielt: {reason}")
        return "\n".join(out)
    out.append(f"Alle Teilnehmer in einem Modell, jeder gegen jeden, {ladder['games_per_pair']} Spiele je "
               f"Paarung (Einzel-Ladders der Lauf-Ordner sind nicht vergleichbar und werden nicht benutzt).")
    out += ["", "| Teilnehmer | mu | sigma | mu-3σ |", "|---|---|---|---|"]
    for label, r in sorted(ladder["ratings"].items(), key=lambda kv: -kv[1]["conservative"]):
        out.append(f"| {label} | {r['mu']:.2f} | {r['sigma']:.2f} | {r['conservative']:.2f} |")
    return "\n".join(out)


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
        changes = config_changes(s, baseline)
        if changes and len(changes) > 1:
            notes.append(f"**Bündel aus {len(changes)} Änderungen** ({', '.join(changes)}): Ein Effekt ist keinem "
                         f"einzelnen Wert zuzuordnen; nur bei schlechtem oder unklarem Ergebnis aufteilen (AUDIT.md §7.6)")
        ds = duel_summary(s.get("duel_baseline"))
        if ds and not math.isnan(ds["goal_diff_ci_low"]):
            m, low, high = ds["goal_diff"], ds["goal_diff_ci_low"], ds["goal_diff_ci_high"]
            span = f"{m:+.3f} Tore/Spiel, 95-%-KI [{low:+.3f}, {high:+.3f}]"
            if low > 0:
                notes.append(f"Hauptkriterium Duell: **besser** als Baseline-Ende ({span}, ganz über 0)")
            elif high < 0:
                notes.append(f"Hauptkriterium Duell: **schlechter** als Baseline-Ende ({span}, ganz unter 0)")
            else:
                notes.append(f"Hauptkriterium Duell: **im Rauschen** ({span} enthält 0); mehr Duell-Spiele "
                             f"oder längerer Lauf")
        elif ds:
            notes.append("Hauptkriterium Duell ohne Einzelspiele (altes duel.exe): kein Intervall")
        else:
            notes.append("Hauptkriterium Duell fehlt (kein duel_end_vs_baseline.json; -Baseline bei run_experiment.ps1?)")
        g, gb = last.get("ep_end_goal"), base_last.get("ep_end_goal")
        if g is not None and gb is not None and gb > 0:
            notes.append(f"Tor-Anteil {g:.3f} gegen {gb:.3f} ({(g - gb) / gb:+.1%})")
        e = last.get("entropy")
        if e is not None and e < 2.5:
            notes.append(f"Entropie {e:.2f} < 2,5: Kollaps-Schwelle aus AUDIT.md H2")
        sps, spsb = last.get("sps"), base_last.get("sps")
        if sps and spsb and abs(sps / spsb - 1) > 0.07:
            notes.append(f"SPS {sps / spsb - 1:+.1%} gegenüber Baseline (über der 7-%-Streuung)")
        out.append(f"* **{s['name']}**: " + "; ".join(notes))
    return "\n".join(out)


def changes_section(experiments: list[dict], baseline: dict) -> str:
    """Was jedes Experiment gegenüber der Baseline ändert (aus den Configs, nicht aus dem Namen)."""
    out = ["", "## Änderungen gegenüber der Baseline (aus config.json)", ""]
    for s in experiments:
        if s is baseline:
            continue
        changes = config_changes(s, baseline)
        if changes is None:
            out.append(f"* **{s['name']}**: config.json fehlt, Änderungen unbekannt")
            continue
        kind = f"Bündel, {len(changes)} Werte" if len(changes) > 1 else f"{len(changes)} Wert"
        detail = ", ".join(f"`{k}` {a} → {b}" for k, (a, b) in changes.items()) or "keine"
        out.append(f"* **{s['name']}** ({kind}): {detail}")
    return "\n".join(out)


def expand_folders(args: list[str]) -> list[Path]:
    """Glob-Muster selbst auflösen (Review-Befund R17): Windows PowerShell reicht results/exp_* an
    native Programme wörtlich weiter. Aus Mustern werden nur Ordner mit summary.json genommen
    (results/exp_*.zip und halbe Ordner fallen heraus); ein Muster ohne Treffer ist ein Fehler.
    Explizit genannte Ordner bleiben, wie sie sind (fehlt summary.json, bricht load_experiment ab)."""
    out: list[Path] = []
    for arg in args:
        if any(ch in arg for ch in "*?["):
            matches = [Path(m) for m in sorted(glob.glob(arg))]
            dirs = [m for m in matches if m.is_dir() and (m / "summary.json").exists()]
            skipped = [m.name for m in matches if m not in dirs]
            if skipped:
                print(f"compare.py: übersprungen (kein Ergebnisordner): {', '.join(skipped)}", file=sys.stderr)
            if not dirs:
                raise SystemExit(f"Muster {arg} trifft keinen Ergebnisordner mit summary.json")
            out += dirs
        else:
            out.append(Path(arg))
    unique: list[Path] = []
    for p in out:
        if all(p.resolve() != q.resolve() for q in unique):
            unique.append(p)
    return unique


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from eval.ladder import DUEL_EXE  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="Ergebnisordner oder Muster wie results/exp_* (löst compare.py selbst auf)")
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="Markdown zusätzlich in Datei schreiben")
    ap.add_argument("--ladder-games", type=int, default=100,
                    help="Spiele je Paarung in der gemeinsamen Ladder (0 = keine Ladder)")
    ap.add_argument("--exe", type=Path, default=DUEL_EXE, help="Pfad zu duel.exe")
    ap.add_argument("--ladder-out", type=Path, default=None,
                    help="Ergebnis der gemeinsamen Ladder (Standard: joint_ladder.json neben --out)")
    a = ap.parse_args()

    experiments = [load_experiment(f) for f in expand_folders(a.folders)]
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

    ladder, reason = None, None
    participants = ladder_participants(experiments, baseline)
    if a.ladder_games <= 0:
        reason = "--ladder-games 0"
    elif len(participants) < 2:
        reason = f"weniger als zwei Checkpoints gefunden ({', '.join(participants) or 'keine'})"
    elif not a.exe.exists():
        reason = f"duel.exe fehlt: {a.exe}"
    else:
        ladder_out = a.ladder_out or ((a.out.parent / "joint_ladder.json") if a.out else None)
        ladder = run_joint_ladder(participants, a.ladder_games, a.exe, ladder_out)

    md = "# Experiment-Vergleich (letztes Fünftel der Iterationen)\n\n"
    md += f"Baseline: `{baseline['_folder']}`\n\n"
    md += ("Hauptkriterium: Duell gegen das Baseline-Ende, mittlere Tordifferenz pro Spiel mit "
           "95-%-t-Intervall (Spiele à 300 s). Dazu Gewinnrate (Remis = halber Sieg, Wilson) und "
           "Tore pro Minute. TrueSkill nur aus der gemeinsamen Ladder unten.\n\n")
    md += build_table(experiments, baseline, ladder) + "\n"
    md += ladder_section(ladder, reason) + "\n"
    md += changes_section(experiments, baseline) + "\n"
    md += verdict_hints(experiments, baseline) + "\n"
    print(md)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
