"""Die Python-Inferenz im Deployment muss dieselben Aktionen liefern wie der C++-Code.

Geprüft wird gegen dump_policy_actions.exe, das die Policy mit libtorch lädt und rechnet.
Ohne Checkpoint oder ohne gebautes Tool werden die Tests übersprungen.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from deploy.policy import export_policy, load_policy

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "obs_golden.json"


def _build_exe(name: str) -> Path:
    """Pfad eines C++-Tools; Build-Ordner per RLBOT_BUILD_DIR überschreibbar (Linux-Build ohne .exe)."""
    build = Path(os.environ.get("RLBOT_BUILD_DIR", str(ROOT / "build" / "cpp_cu128")))
    exe = build / f"{name}.exe"
    return exe if exe.exists() or not (build / name).exists() else build / name


DUMPER = _build_exe("dump_policy_actions")
# Audit M5: nur der Hauptlauf, numerisch sortiert (vorher lexikografisch über alle Läufe,
# also auch über runs/archive/* mit anderer Netzgröße). Überschreibbar per Umgebungsvariable.
PARITY_RUN = Path(os.environ.get("RLBOT_PARITY_RUN", str(ROOT / "runs" / "lucy_1v1")))


def _find_checkpoint(run_dir: Path = PARITY_RUN) -> Path | None:
    folder = run_dir / "checkpoints" if (run_dir / "checkpoints").exists() else run_dir
    candidates = sorted(folder.glob("*/PPO_POLICY.lt"), key=lambda p: int(p.parent.name))
    return candidates[-1] if candidates else None


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    path = _find_checkpoint()
    if path is None:
        pytest.skip("kein Trainings-Checkpoint vorhanden")
    return path


@pytest.fixture(scope="module")
def obs_batch() -> np.ndarray:
    if not FIXTURE.exists():
        pytest.skip("Golden-Fixtures fehlen")
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rows = [p["obs"] for case in data["cases"] for p in case["players"]]
    return np.array(rows, dtype=np.float32)


def test_checkpoint_matches_obs_layout(checkpoint, obs_batch):
    policy = load_policy(checkpoint)
    assert policy.meta.obs_size == obs_batch.shape[1], (
        "Die trainierte Policy erwartet eine andere Obs-Größe als der Obs-Builder liefert"
    )
    assert policy.meta.action_count == 90


def test_probs_and_actions_match_cpp(checkpoint, obs_batch):
    if not DUMPER.exists():
        pytest.skip("dump_policy_actions.exe nicht gebaut")

    policy = load_policy(checkpoint)
    meta = policy.meta

    with tempfile.TemporaryDirectory() as tmp:
        obs_path = Path(tmp) / "obs.json"
        out_path = Path(tmp) / "out.json"
        obs_path.write_text(json.dumps(obs_batch.tolist()), encoding="utf-8")

        result = subprocess.run(
            [str(DUMPER), str(checkpoint), str(meta.obs_size), str(meta.action_count),
             ",".join(map(str, meta.layer_sizes)), str(obs_path), str(out_path)],
            capture_output=True, text=True, cwd=ROOT,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        cpp = json.loads(out_path.read_text(encoding="utf-8"))

    cpp_probs = np.array(cpp["probs"], dtype=np.float64)
    cpp_actions = np.array(cpp["actions"], dtype=np.int64)

    py_probs = policy.action_probs(obs_batch).astype(np.float64)
    py_actions = np.asarray(policy.act(obs_batch, deterministic=True))

    assert py_probs.shape == cpp_probs.shape
    max_diff = float(np.abs(py_probs - cpp_probs).max())
    assert max_diff < 1e-5, f"Wahrscheinlichkeiten weichen ab, größte Differenz {max_diff:.2e}"
    assert np.array_equal(py_actions, cpp_actions), (
        f"{int((py_actions != cpp_actions).sum())} von {len(cpp_actions)} Aktionen unterschiedlich"
    )
    print(f"\n{len(cpp_actions)} Obs geprüft, größte Wahrscheinlichkeitsdifferenz: {max_diff:.2e}")


def test_export_roundtrip_keeps_outputs(checkpoint, obs_batch, tmp_path):
    original = load_policy(checkpoint)
    out = tmp_path / "policy.pt"
    export_policy(checkpoint, out)
    reloaded = load_policy(out)

    assert reloaded.meta.layer_sizes == original.meta.layer_sizes
    before = original.action_probs(obs_batch)
    after = reloaded.action_probs(obs_batch)
    assert np.array_equal(before, after), "Export verändert die Ausgaben"
    assert out.with_suffix(".json").exists(), "Metadaten-JSON fehlt"
