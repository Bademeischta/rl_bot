"""Wählt den neuesten VOLLSTÄNDIGEN Checkpoint eines Laufs (Review-Befund R16).

    python tools/experiments/pick_checkpoint.py runs/exp_x/checkpoints [--min-steps N]

Vorher nahm run_experiment.ps1 einfach den Ordner mit der höchsten Step-Zahl. Wurde der Trainer
mitten im Speichern beendet (Notfall-Kill, Absturz, Stromausfall), war das ein halb geschriebener
Checkpoint, und Duell, Ladder und Zusammenfassung liefen auf Müll oder brachen ab.

Vollständig heißt:
  1. alle fünf Dateien vorhanden und nicht leer (Policy, Critic, beide Optimizer, RUNNING_STATS.json)
  2. die vier .lt-Dateien sind intakte Zip-Archive (Zentralverzeichnis vorhanden, CRC aller
     Einträge stimmt); ein abgebrochener Schreibvorgang hinterlässt das nicht
  3. RUNNING_STATS.json ist gültiges JSON und cumulative_timesteps == Ordnername
  4. Ladeprüfung: PPO_POLICY.lt lädt mit deploy.policy.load_policy (derselbe Weg wie Deployment
     und Paritätstests)

Ausgabe: Pfad des gewählten Checkpoints auf stdout, verworfene Kandidaten mit Grund auf stderr.
Exit 0 = gefunden, 1 = kein vollständiger Checkpoint (mit --min-steps: keiner ab dieser Step-Zahl).
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ("PPO_POLICY.lt", "PPO_CRITIC.lt", "PPO_POLICY_OPTIM.lt", "PPO_CRITIC_OPTIM.lt", "RUNNING_STATS.json")


def check_checkpoint(folder: Path, load: bool = True) -> str | None:
    """None, wenn der Checkpoint vollständig ist, sonst der Grund."""
    for name in REQUIRED:
        f = folder / name
        if not f.exists():
            return f"{name} fehlt"
        if f.stat().st_size == 0:
            return f"{name} ist leer"
    for name in REQUIRED[:4]:
        try:
            with zipfile.ZipFile(folder / name) as z:
                bad = z.testzip()
        except (zipfile.BadZipFile, OSError) as e:
            return f"{name} ist kein intaktes Archiv ({e})"
        if bad is not None:
            return f"{name}: Prüfsumme falsch in {bad}"
    try:
        stats = json.loads((folder / "RUNNING_STATS.json").read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        return f"RUNNING_STATS.json unlesbar ({e})"
    if str(stats.get("cumulative_timesteps")) != folder.name:
        return f"RUNNING_STATS.json: cumulative_timesteps {stats.get('cumulative_timesteps')} passt nicht zu {folder.name}"
    if load:
        sys.path.insert(0, str(ROOT))
        try:
            from deploy.policy import load_policy
            load_policy(folder / "PPO_POLICY.lt")
        except Exception as e:  # noqa: BLE001 - jeder Ladefehler heißt: unbrauchbar
            return f"PPO_POLICY.lt lädt nicht ({type(e).__name__}: {e})"
    return None


def pick_latest_complete(checkpoints: Path, min_steps: int = -1, load: bool = True) -> tuple[Path | None, list[str]]:
    """(neuester vollständiger Checkpoint oder None, Liste verworfener Kandidaten mit Grund)."""
    candidates = sorted((d for d in checkpoints.iterdir() if d.is_dir() and d.name.isdigit()),
                        key=lambda d: int(d.name), reverse=True)
    rejected = []
    for folder in candidates:
        if int(folder.name) < min_steps:
            break
        reason = check_checkpoint(folder, load)
        if reason is None:
            return folder, rejected
        rejected.append(f"{folder.name}: {reason}")
    return None, rejected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", type=Path, help="Ordner mit <steps>-Unterordnern")
    ap.add_argument("--min-steps", type=int, default=-1, help="ältere Checkpoints nicht in Betracht ziehen")
    ap.add_argument("--no-load", action="store_true", help="ohne Ladeprüfung (schneller, ohne torch)")
    a = ap.parse_args()
    if not a.checkpoints.is_dir():
        print(f"Ordner fehlt: {a.checkpoints}", file=sys.stderr)
        return 1
    chosen, rejected = pick_latest_complete(a.checkpoints, a.min_steps, not a.no_load)
    for line in rejected:
        print(f"verworfen: {line}", file=sys.stderr)
    if chosen is None:
        print("kein vollständiger Checkpoint gefunden", file=sys.stderr)
        return 1
    print(chosen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
