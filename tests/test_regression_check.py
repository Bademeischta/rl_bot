"""Regressions-Check (Hauptlauf-Betrieb B2): tools/regression_check.py über echte Checkpoints und duel.exe.

Die Checkpoints werden in einen Test-Lauf KOPIERT (runs/lucy_1v1 wird nur gelesen); geprüft wird die
Auswahl (neuester vollständiger, ~gap älterer), dass die Quelle unverändert bleibt, das Urteil und der
Verlauf mit Datum.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import regression_check as rc  # noqa: E402

MAIN_CKPTS = ROOT / "runs" / "lucy_1v1" / "checkpoints"
DUEL = ROOT / "build" / "cpp_cu128" / "duel.exe"


def _real() -> list[Path]:
    if not MAIN_CKPTS.is_dir():
        return []
    return sorted((d for d in MAIN_CKPTS.iterdir() if d.is_dir() and d.name.isdigit()
                   and (d / "PPO_POLICY.lt").exists()), key=lambda d: int(d.name))


def _digest(folder: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(folder.iterdir()):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def test_verdict_follows_the_confidence_interval():
    assert rc.verdict(0.05, 0.4) == "besser"
    assert rc.verdict(-0.4, -0.02) == "schlechter"
    assert rc.verdict(-0.1, 0.2) == "gleich"


@pytest.mark.skipif(len(_real()) < 3, reason="echte Checkpoints fehlen")
def test_pick_pair_takes_the_newest_complete_and_one_about_gap_older(tmp_path):
    real = _real()
    newest, older, oldest = real[-1], real[len(real) // 2], real[0]
    run = tmp_path / "run"
    for c in (oldest, older, newest):
        shutil.copytree(c, run / "checkpoints" / c.name)
    # halb geschriebener neuerer Checkpoint (Trainer schreibt gerade): wird übersprungen
    broken = run / "checkpoints" / str(int(newest.name) + 25_000_000)
    shutil.copytree(newest, broken)
    (broken / "PPO_POLICY_OPTIM.lt").write_bytes(b"")
    gap = int(newest.name) - int(older.name)
    n, o = rc.pick_pair(run, gap, 10_000_000)
    assert (n.name, o.name) == (newest.name, older.name)
    # kein Checkpoint in Reichweite -> klarer Abbruch statt eines falschen Gegners
    with pytest.raises(SystemExit, match="kein vollständiger Checkpoint nahe"):
        rc.pick_pair(run, gap * 10, 10_000_000)


@pytest.mark.skipif(len(_real()) < 2 or not DUEL.exists(), reason="echte Checkpoints oder duel.exe fehlen")
def test_regression_check_end_to_end_writes_verdict_and_history_and_leaves_the_run_untouched(tmp_path):
    real = _real()
    newest, old = real[-1], real[0]
    run = tmp_path / "lucy_test"
    for c in (old, newest):
        shutil.copytree(c, run / "checkpoints" / c.name)
    before = {c.name: _digest(run / "checkpoints" / c.name) for c in (old, newest)}
    out_root = tmp_path / "regression"
    gap = int(newest.name) - int(old.name)
    args = ["--run", str(run), "--gap", str(gap), "--games", "16", "--threads", "2", "--kickoffs", "16",
            "--exe", str(DUEL), "--out-root", str(out_root)]
    code = rc.main(args)
    assert code in (0, 3)
    assert {c: _digest(run / "checkpoints" / c) for c in before} == before      # nur gelesen

    [folder] = [p for p in out_root.iterdir() if p.is_dir()]
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    assert (result["neu"], result["alt"], result["spiele"]) == (int(newest.name), int(old.name), 16)
    assert result["urteil"] == rc.verdict(result["tordiff_ki_low"], result["tordiff_ki_high"])
    assert (code == 3) == (result["urteil"] == "schlechter")
    assert 0 <= result["anstoss_zuerst_neu"] <= 1 and result["anstoss_zeit_neu"] > 0
    duel = json.loads((folder / "duel.json").read_text(encoding="utf-8"))
    assert duel["threads"] == 2 and duel["games"] == 16

    # Verlauf: eine Zeile je Lauf, mit Datum; ein zweiter Lauf hängt an
    rc.main(args[:-2] + ["--out-root", str(out_root)])
    rows = list(csv.DictReader((out_root / "lucy_test_history.csv").open(encoding="utf-8")))
    assert len(rows) == 2 and all(r["datum"] and r["urteil"] in ("besser", "gleich", "schlechter") for r in rows)
    md = (out_root / "lucy_test_history.md").read_text(encoding="utf-8")
    assert md.count(f"| {int(newest.name):,} | {int(old.name):,} |") == 2
