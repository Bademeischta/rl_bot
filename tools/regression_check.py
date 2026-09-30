"""Regressions-Check für einen laufenden Trainingslauf (Hauptlauf-Betrieb B2).

    python tools/regression_check.py [--run runs/lucy_1v1] [--gap 1000000000] [--games 1000]
                                     [--threads 2] [--kickoffs 500]

Nimmt den neuesten VOLLSTÄNDIGEN Checkpoint des Laufs und den vollständigen Checkpoint, der am nächsten
an "neuester − gap" liegt, und prüft, ob der neue Stand besser, gleich oder schlechter spielt:

* Duell (duel.exe, 1000 Spiele à 300 s, beide Seiten ziehen wie im Training): mittlere Tordifferenz
  neu − alt pro Spiel mit 95-%-Intervall, Gewinnrate, Anstoß-Kennzahlen je Seite.
* Anstöße (eval/kickoff_eval.py, Bot-Code): Anteil "zuerst am Ball", Zeit bis zur Berührung.
* Urteil: **besser**, wenn das Intervall der Tordifferenz ganz über 0 liegt, **schlechter**, wenn ganz
  darunter, sonst **gleich**.

Läuft neben dem Training: duel.exe bekommt `--threads` (Default 2, je Thread ein Kern), die
Anstoß-Auswertung einen Torch-Thread. Der Lauf wird nur gelesen: Die beiden PPO_POLICY.lt werden zuerst
in den Ergebnisordner kopiert (die Checkpoint-Rotation des Trainers darf die Quelle danach löschen),
halb geschriebene Checkpoints erkennt tools/experiments/pick_checkpoint.py.

Ausgabe: results/regression/<lauf>_<datum>_<neu>_vs_<alt>/ (duel.json, kickoff.json, result.json) und je
eine Zeile in results/regression/<lauf>_history.csv und <lauf>_history.md (Verlauf mit Datum).
Exit 0 = besser oder gleich, 3 = schlechter, 1 = Fehler (z. B. kein passender alter Checkpoint).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "experiments"))
from metrics_util import duel_stats  # noqa: E402
from pick_checkpoint import check_checkpoint  # noqa: E402

HISTORY_FIELDS = [
    "datum", "lauf", "neu", "alt", "abstand", "urteil", "tordiff", "tordiff_ki_low", "tordiff_ki_high",
    "gewinnrate", "siege_neu", "siege_alt", "remis", "spiele", "tore_pro_min",
    "anstoss_zuerst_neu", "anstoss_zuerst_ki_low", "anstoss_zuerst_ki_high", "anstoss_zeit_neu", "anstoss_zeit_alt",
    "duell_anstoss_zuerst_neu", "duell_anstoss_tore_neu", "duell_anstoss_tore_alt", "sekunden",
]


def checkpoints(run: Path) -> list[Path]:
    folder = run / "checkpoints"
    dirs = [d for d in folder.iterdir() if d.is_dir() and d.name.isdigit()] if folder.is_dir() else []
    return sorted(dirs, key=lambda d: int(d.name))


def pick_pair(run: Path, gap: int, max_deviation: int, load: bool = True) -> tuple[Path, Path]:
    """(neuester vollständiger, vollständiger nächst an neuester − gap). SystemExit, wenn es keinen gibt."""
    cands = checkpoints(run)
    newest = None
    for c in reversed(cands):
        reason = check_checkpoint(c, load=load)
        if reason is None:
            newest = c
            break
        print(f"übersprungen {c.name}: {reason}", file=sys.stderr)
    if newest is None:
        raise SystemExit(f"kein vollständiger Checkpoint in {run / 'checkpoints'}")
    target = int(newest.name) - gap
    older = sorted((c for c in cands if int(c.name) < int(newest.name)), key=lambda c: abs(int(c.name) - target))
    for c in older:
        if abs(int(c.name) - target) > max_deviation:
            break
        if check_checkpoint(c, load=load) is None:
            return newest, c
    raise SystemExit(f"kein vollständiger Checkpoint nahe {target:,} Steps (neuester {int(newest.name):,}, "
                     f"Abstand {gap:,} ± {max_deviation:,}); ältester vorhandener: "
                     f"{int(cands[0].name):,}" if cands else "keine Checkpoints")


def verdict(low: float, high: float) -> str:
    if low > 0:
        return "besser"
    if high < 0:
        return "schlechter"
    return "gleich"


def run_checked(cmd: list[str], env: dict, log: Path) -> None:
    with log.open("w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f"{Path(cmd[0]).name} Exit {rc}, siehe {log}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, default=ROOT / "runs" / "lucy_1v1")
    ap.add_argument("--gap", type=int, default=1_000_000_000, help="Abstand in Steps zum alten Checkpoint")
    ap.add_argument("--max-deviation", type=int, default=250_000_000,
                    help="so weit darf der alte Checkpoint von neuester − gap abweichen")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=2, help="Threads für duel.exe (je Thread ein Kern)")
    ap.add_argument("--kickoffs", type=int, default=500, help="0 = keine Anstoß-Auswertung")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--exe", type=Path, default=ROOT / "build" / "cpp_cu128" / "duel.exe")
    ap.add_argument("--out-root", type=Path, default=ROOT / "results" / "regression")
    ap.add_argument("--no-load-check", action="store_true", help="(Tests) Checkpoints nicht per Torch laden")
    a = ap.parse_args(argv)

    run = a.run.resolve()
    if not a.exe.exists():
        raise SystemExit(f"duel.exe fehlt: {a.exe}")
    newest, old = pick_pair(run, a.gap, a.max_deviation, load=not a.no_load_check)
    started = dt.datetime.now()
    stamp = started.strftime("%Y-%m-%d_%H%M%S")
    out = a.out_root / f"{run.name}_{stamp}_{newest.name}_vs_{old.name}"
    out.mkdir(parents=True, exist_ok=False)
    # Nur lesen: Policies kopieren, damit die Rotation des Trainers während des Duells nichts wegnimmt
    pol_new = out / f"policy_{newest.name}.lt"
    pol_old = out / f"policy_{old.name}.lt"
    shutil.copyfile(newest / "PPO_POLICY.lt", pol_new)
    shutil.copyfile(old / "PPO_POLICY.lt", pol_old)
    print(f"neu {int(newest.name):,} gegen alt {int(old.name):,} (Abstand {int(newest.name) - int(old.name):,} Steps), "
          f"{a.games} Spiele mit {a.threads} Threads, {a.kickoffs} Anstöße -> {out}", flush=True)

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    duel_json = out / "duel.json"
    run_checked([str(a.exe), "--a", str(pol_new), "--b", str(pol_old), "--games", str(a.games),
                 "--threads", str(a.threads), "--seed", str(a.seed), "--meshes", str(ROOT / "collision_meshes"),
                 "--out", str(duel_json)], env, out / "duel.log")
    d = json.loads(duel_json.read_text(encoding="utf-8"))
    s = duel_stats(d)
    sa, sb = d.get("stats", {}).get("a", {}), d.get("stats", {}).get("b", {})

    ko = None
    if a.kickoffs > 0:
        ko_json = out / "kickoff.json"
        run_checked([sys.executable, str(ROOT / "eval" / "kickoff_eval.py"), "--a", f"policy:{pol_new}",
                     "--b", f"policy:{pol_old}", "--kickoffs", str(a.kickoffs), "--seed", str(a.seed),
                     "--out", str(ko_json)], env, out / "kickoff.log")
        ko = json.loads(ko_json.read_text(encoding="utf-8"))["summary"]["all"]

    result = {
        "datum": started.strftime("%Y-%m-%d %H:%M"), "lauf": run.name, "neu": int(newest.name), "alt": int(old.name),
        "abstand": int(newest.name) - int(old.name),
        "urteil": verdict(s["goal_diff_ci_low"], s["goal_diff_ci_high"]),
        "tordiff": round(s["goal_diff"], 4), "tordiff_ki_low": round(s["goal_diff_ci_low"], 4),
        "tordiff_ki_high": round(s["goal_diff_ci_high"], 4), "gewinnrate": round(s["win_rate"], 4),
        "siege_neu": d.get("wins_a"), "siege_alt": d.get("wins_b"), "remis": d.get("draws"), "spiele": s["games"],
        "tore_pro_min": round(s["goals_per_minute"], 4),
        "anstoss_zuerst_neu": round(ko["a_first_rate"], 4) if ko else "",
        "anstoss_zuerst_ki_low": round(ko["a_first_ci"][0], 4) if ko else "",
        "anstoss_zuerst_ki_high": round(ko["a_first_ci"][1], 4) if ko else "",
        "anstoss_zeit_neu": round(ko["a_time_to_touch"], 3) if ko else "",
        "anstoss_zeit_alt": round(ko["b_time_to_touch"], 3) if ko else "",
        "duell_anstoss_zuerst_neu": round(sa.get("kickoff_first_touch_rate", float("nan")), 4),
        "duell_anstoss_tore_neu": sa.get("kickoff_goals_10s", ""), "duell_anstoss_tore_alt": sb.get("kickoff_goals_10s", ""),
        "sekunden": round((dt.datetime.now() - started).total_seconds(), 1),
    }
    (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

    hist_csv = a.out_root / f"{run.name}_history.csv"
    new_file = not hist_csv.exists()
    with hist_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(result)
    line = (f"| {result['datum']} | {result['neu']:,} | {result['alt']:,} | **{result['urteil']}** | "
            f"{result['tordiff']:+.3f} [{result['tordiff_ki_low']:+.3f}; {result['tordiff_ki_high']:+.3f}] | "
            f"{result['siege_neu']}:{result['siege_alt']} ({result['remis']} remis) | "
            + (f"{result['anstoss_zuerst_neu']:.1%} [{result['anstoss_zuerst_ki_low']:.1%}; "
               f"{result['anstoss_zuerst_ki_high']:.1%}], {result['anstoss_zeit_neu']} s gegen {result['anstoss_zeit_alt']} s"
               if ko else "-")
            + " |")
    hist_md = a.out_root / f"{run.name}_history.md"
    if not hist_md.exists():
        hist_md.write_text(
            f"# Regressions-Checks `{run.name}`\n\nNeuester Checkpoint gegen einen ~{a.gap:,} Steps älteren "
            "(tools/regression_check.py). Urteil nach dem 95-%-Intervall der Tordifferenz neu − alt.\n\n"
            "| Datum | neu | alt | Urteil | Tordifferenz/Spiel [95-%-KI] | Siege neu:alt | Anstoß zuerst (neu) [KI], Zeit neu/alt |\n"
            "|---|---|---|---|---|---|---|\n", encoding="utf-8")
    with hist_md.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

    print(f"Urteil: neuer Stand {result['urteil'].upper()} "
          f"(Tordifferenz {result['tordiff']:+.3f} [{result['tordiff_ki_low']:+.3f}; {result['tordiff_ki_high']:+.3f}] "
          f"pro Spiel, Siege {result['siege_neu']}:{result['siege_alt']})")
    if ko:
        print(f"Anstöße: neu zuerst {result['anstoss_zuerst_neu']:.1%}, Berührung {result['anstoss_zeit_neu']} s "
              f"gegen {result['anstoss_zeit_alt']} s")
    print(f"Verlauf: {hist_md}")
    return 3 if result["urteil"] == "schlechter" else 0


if __name__ == "__main__":
    sys.exit(main())
