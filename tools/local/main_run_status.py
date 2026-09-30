"""Stand des Hauptlaufs mit Grenzwerten (Hauptlauf-Betrieb B4).

    python tools/local/main_run_status.py [--run runs/lucy_1v1] [--window 200] [--min-sps 150000] [--log <csv>]

Mittelt die letzten `--window` Iterationen aus metrics.csv (nur Iterationen dieses Laufs: ab dem letzten
Start laut config_used.json/_started ist nicht nachvollziehbar, deshalb einfach die letzten Zeilen) und
prüft die Grenzwerte aus dem Betriebsauftrag (29.09./30.09.2026):

  Entropie > 2,5 · KL < 0,01 · Clip-Fraction < 0,10 · Value Loss endlich und ohne Sprung
  (Mittel ≤ 3 × Mittel des Fensters davor) · keine nan/inf in den Kernspalten ·
  Tor-Anteil der Episodenenden ohne Anstoß-Drill ≥ 0,95 (ep_end_goal / (1 − ep_end_drill); mit Drill
  enden ~36 % der Episoden planmäßig nach 6 s, deshalb ist ep_end_goal selbst ~0,64) ·
  Zeit-/NoTouch-Timeouts ≤ 0,01 · SPS ≥ --min-sps (Spiele nebenbei senken die SPS: dann nur Hinweis).

Dazu: neuester Checkpoint, freier Plattenplatz, läuft train_bot.exe. Mit --log wird eine Zeile an eine
CSV angehängt (Verlauf der Beobachtung). Exit 0 = alles im Rahmen, 3 = Grenzwert verletzt, 1 = Fehler.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "experiments"))
from metrics_util import read_rows, to_float  # noqa: E402

CORE = ["Policy Entropy", "Mean KL Divergence", "SB3 Clip Fraction", "Value Function Loss",
        "Avg Advantage", "Avg Val Target"]


def mean(rows: list[dict], key: str) -> float:
    vals = [to_float(r.get(key)) for r in rows]
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def nz(v: float) -> float:
    """nan (Spalte fehlt im Fenster) als 0."""
    return 0.0 if math.isnan(v) else v


def non_finite(rows: list[dict], key: str) -> int:
    n = 0
    for r in rows:
        raw = r.get(key)
        if raw in (None, ""):
            continue
        v = to_float(raw)
        n += math.isnan(v) or math.isinf(v)
    return n


def evaluate(rows: list[dict], window: int, min_sps: float) -> dict:
    last = rows[-window:]
    prev = rows[-2 * window:-window]
    s: dict = {"iterationen": len(last)}
    s["steps"] = int(to_float(last[-1].get("Cumulative Timesteps"))) if last else 0
    for key, name in [("Policy Entropy", "entropie"), ("Mean KL Divergence", "kl"), ("SB3 Clip Fraction", "clip"),
                      ("Value Function Loss", "value_loss"), ("Overall Steps/Second", "sps"),
                      ("ep_end_goal", "ep_end_goal"), ("ep_end_drill", "ep_end_drill"), ("ep_end_time", "ep_end_time"),
                      ("ep_end_notouch", "ep_end_notouch"), ("Timesteps Collected", "iteration_steps"),
                      ("Steps Collected During Learn", "steps_waehrend_lernen")]:
        s[name] = mean(last, key)
    drill = nz(s["ep_end_drill"])
    s["tor_anteil_ohne_drill"] = s["ep_end_goal"] / (1 - drill) if drill < 1 else math.nan
    s["value_loss_davor"] = mean(prev, "Value Function Loss") if prev else math.nan
    bad = {k: non_finite(last, k) for k in CORE}
    s["nicht_endlich"] = sum(bad.values())

    checks = [
        ("Entropie > 2,5", s["entropie"] > 2.5),
        ("KL < 0,01", s["kl"] < 0.01),
        ("Clip-Fraction < 0,10", s["clip"] < 0.10),
        ("keine nan/inf (" + ", ".join(k for k, v in bad.items() if v) + ")" if s["nicht_endlich"] else "keine nan/inf",
         s["nicht_endlich"] == 0),
        ("Value Loss stabil (≤ 3× Fenster davor)",
         math.isfinite(s["value_loss"]) and (math.isnan(s["value_loss_davor"]) or s["value_loss"] <= 3 * s["value_loss_davor"])),
        ("Tor-Anteil ohne Drill ≥ 0,95", s["tor_anteil_ohne_drill"] >= 0.95),
        ("Timeouts ≤ 0,01", nz(s["ep_end_time"]) + nz(s["ep_end_notouch"]) <= 0.01),
    ]
    s["checks"] = checks
    s["verletzt"] = [name for name, ok in checks if not ok]
    s["sps_niedrig"] = not (s["sps"] >= min_sps)
    return s


def trainer_running() -> bool:
    if os.name != "nt":
        return False
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq train_bot.exe", "/NH"], capture_output=True, text=True,
                         errors="replace").stdout
    return "train_bot.exe" in out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "runs" / "lucy_1v1")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--min-sps", type=float, default=150_000)
    ap.add_argument("--log", type=Path, default=None, help="Zeile an diese CSV anhängen")
    a = ap.parse_args(argv)

    metrics = a.run / "metrics.csv"
    if not metrics.exists():
        print(f"FEHLER: {metrics} fehlt")
        return 1
    rows = read_rows(metrics)
    if len(rows) < a.window:
        print(f"FEHLER: nur {len(rows)} Iterationen in {metrics}")
        return 1
    s = evaluate(rows, a.window, a.min_sps)
    ckpts = sorted((d for d in (a.run / "checkpoints").iterdir() if d.is_dir() and d.name.isdigit()),
                   key=lambda d: int(d.name)) if (a.run / "checkpoints").is_dir() else []
    free_gb = shutil.disk_usage(a.run).free / 1e9
    running = trainer_running()
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"Stand {now}: {s['steps']:,} Steps, letzte {s['iterationen']} Iterationen gemittelt; "
          f"train_bot.exe {'läuft' if running else 'läuft NICHT'}")
    print(f"  SPS {s['sps']:,.0f} · Entropie {s['entropie']:.3f} · KL {s['kl']:.4f} · Clip {s['clip']:.3f} · "
          f"Value Loss {s['value_loss']:.3f} (davor {s['value_loss_davor']:.3f})")
    print(f"  ep_end_goal {s['ep_end_goal']:.3f} (Drill {s['ep_end_drill']:.3f}, Tor-Anteil ohne Drill "
          f"{s['tor_anteil_ohne_drill']:.3f}) · Zeit-Timeout {s['ep_end_time']:.4f} · NoTouch {s['ep_end_notouch']:.4f}")
    print(f"  Iteration {s['iteration_steps']:,.0f} Steps, davon während der Lernphase {s['steps_waehrend_lernen']:,.0f}")
    print(f"  Checkpoints {len(ckpts)}" + (f" ({int(ckpts[0].name):,} bis {int(ckpts[-1].name):,})" if ckpts else "")
          + f" · frei auf der Platte {free_gb:.1f} GB")
    for name, ok in s["checks"]:
        print(f"  [{'OK' if ok else 'VERLETZT'}] {name}")
    if s["sps_niedrig"]:
        print(f"  [HINWEIS] SPS unter {a.min_sps:,.0f} (läuft nebenher ein Spiel? Das ist kein Fehler)")
    print("Ergebnis: " + ("alle Grenzwerte eingehalten" if not s["verletzt"] else "GRENZWERT VERLETZT: " + "; ".join(s["verletzt"])))

    if a.log:
        new = not a.log.exists()
        a.log.parent.mkdir(parents=True, exist_ok=True)
        with a.log.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["zeit", "steps", "sps", "entropie", "kl", "clip", "value_loss", "ep_end_goal",
                            "tor_anteil_ohne_drill", "timeouts", "steps_waehrend_lernen", "checkpoints", "frei_gb",
                            "laeuft", "verletzt"])
            w.writerow([now, s["steps"], round(s["sps"]), round(s["entropie"], 4), round(s["kl"], 5),
                        round(s["clip"], 4), round(s["value_loss"], 4), round(s["ep_end_goal"], 4),
                        round(s["tor_anteil_ohne_drill"], 4),
                        round(nz(s["ep_end_time"]) + nz(s["ep_end_notouch"]), 5),
                        round(s["steps_waehrend_lernen"]), len(ckpts), round(free_gb, 1), int(running),
                        "; ".join(s["verletzt"])])
    return 3 if s["verletzt"] else 0


if __name__ == "__main__":
    sys.exit(main())
