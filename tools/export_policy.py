"""Exportiert einen Trainings-Checkpoint als deploy-fertige Policy-Datei.

    python tools/export_policy.py runs/lucy_1v1/checkpoints --out deploy/rlbot/policy.pt

Ohne Checkpoint-Angabe wird der neueste Unterordner (höchste Step-Zahl) genommen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from deploy.policy import export_policy, load_policy


def latest_checkpoint(folder: Path) -> Path:
    subdirs = [d for d in folder.iterdir() if d.is_dir() and d.name.isdigit()]
    if not subdirs:
        raise SystemExit(f"Keine Checkpoints in {folder}")
    return max(subdirs, key=lambda d: int(d.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path, help="Checkpoint-Ordner oder Ordner mit Checkpoints")
    ap.add_argument("--out", type=Path, default=Path("deploy/rlbot/policy.pt"))
    a = ap.parse_args()

    path = a.checkpoint
    if path.is_dir() and not (path / "PPO_POLICY.lt").exists():
        path = latest_checkpoint(path)

    meta = export_policy(path, a.out)
    print(f"Quelle:      {path}")
    print(f"Ziel:        {a.out}")
    print(f"Obs-Größe:   {meta.obs_size}")
    print(f"Aktionen:    {meta.action_count}")
    print(f"Schichten:   {meta.layer_sizes}")
    print(f"Steps:       {meta.timesteps:,}")

    # Gegenprobe: Reimport muss dieselbe Architektur ergeben
    reloaded = load_policy(a.out)
    assert reloaded.meta.obs_size == meta.obs_size
    assert reloaded.meta.action_count == meta.action_count
    assert reloaded.meta.layer_sizes == meta.layer_sizes
    print("Reimport geprüft: identisch")


if __name__ == "__main__":
    main()
