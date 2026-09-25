# LOCAL_RUNBOOK: Was lokal auf dem Trainings-PC zu tun ist

Alles hier braucht den Windows-PC (Ryzen 7 8700F, RTX 5070, VS 2026, cu128-libtorch, den
Ordner `runs/`). In der Cloud-Session war das nicht möglich; was dort geprüft wurde, steht in
`AUDIT_PROGRESS.md`. Reihenfolge einhalten: Jeder Schritt setzt den vorherigen voraus.

Alle Skripte sind PowerShell, schreiben ein Log in ihren Ergebnisordner und brechen mit klarer
Meldung ab (`$ErrorActionPreference = 'Stop'`). **Kein Skript löscht oder überschreibt etwas in
`runs/`**; Experimente kopieren den Checkpoint in einen neuen Ordner.

Dauerangaben sind Schätzungen auf Basis der im Audit gelesenen ~68.000 SPS (AUDIT.md §0,
„lokal nachmessen"); die echten Zeiten stehen hinterher in den Logs.

## 0. Vorbereitung (einmalig, ~10 Minuten)

```powershell
cd C:\RLbot
git fetch origin
git checkout main          # nach dem Merge der Review-Fixes; vorher: claude/review-fixes
git pull

# H4: Rückfallpunkt markieren - der Commit, mit dem der Hauptlauf trainiert wurde
git tag baseline-2.7G 54105bf
git push origin baseline-2.7G

# Stufe 0 Punkt 2: aktuellen Checkpoint AUSSERHALB von runs\ sichern (Pfad frei wählen).
# Der Hauptlauf stand beim Review-Fix (25.09.2026) bei 3907335040 Steps; 2704829056 aus dem
# Audit ist durch checkpoints_to_keep = 10 inzwischen gelöscht. Immer den neuesten nehmen:
$C = (Get-ChildItem runs\lucy_1v1\checkpoints -Directory | Sort-Object { [long]$_.Name } | Select-Object -Last 1).FullName
Copy-Item $C "D:\rlbot_backup\$(Split-Path $C -Leaf)" -Recurse

# Python-Umgebung auf die Pins bringen (torch zuerst separat, rlgym-ppo aus git)
.\.venv\Scripts\python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install git+https://github.com/AechPro/rlgym-ppo
```

Der Hauptlauf (`train_bot.exe`) muss für die Schritte 1–4 **gestoppt** sein: Build und Tests
würden ihn bremsen, die Experimente brauchen die GPU. Stoppen kurz nach einem Checkpoint
(`docs/phases.md`, „Pausieren und Fortsetzen").

**Achtung beim Fortsetzen des Hauptlaufs:** Der Neubau ersetzt `build\cpp_cu128\train_bot.exe`.
Die Binaries, mit denen `runs\lucy_1v1` bis 3,9 Mrd. Steps trainiert wurde (ohne K1b, alter
Config-Parser), liegen gesichert in `build\cpp_cu128_vor_review_2026-09-25\`. Der neue Trainer
verhält sich mit `lucy_1v1.json` wie besprochen: `seed_envs` Default false (R6), aber K1b
(Timeouts als Truncation) und K1a (900 s) sind aktiv; Rückweg für K1b:
`"env": { "timeouts_as_truncation": false }`.

Skripte immer mit `powershell` (Windows PowerShell 5.1) starten. Wer eine `.ps1` bearbeitet:
als UTF-8 **mit BOM** speichern (oder nur ASCII verwenden) und native Programme über
`Invoke-Native` (`tools\NativeCommand.ps1`) aufrufen; `tests\test_build_scripts.py` prüft beides.

## 1. Prüfpaket: `tools\local\run_all_checks.ps1` (~20–30 Minuten)

```powershell
powershell -ExecutionPolicy Bypass -File tools\local\run_all_checks.ps1
```

| Schritt | Dauer (geschätzt) | Erfolg erkennbar an |
|---|---|---|
| 1 Branch-Stand | Sekunden | `Branch: claude/rlbot-audit-roadmap-c8t7nl`, keine uncommitteten Änderungen |
| 2 Patches + Build cu128 | 5–15 Minuten (wie ein normaler Build) | `apply_patches.ps1`: 2 Patches „angewendet" (beim zweiten Mal „bereits angewendet"); Build endet ohne `error` |
| 3 Tests, 2 Durchläufe | 2–5 Minuten | `75 bestanden, 0 fehlgeschlagen` (C++), `Golden-Fixtures unveraendert`, Python ≥ 97 `passed` (3 Policy-Paritätstests laufen nur mit Checkpoint, siehe Schritt 6), `Alle Durchläufe bestanden.` |
| 4 Python-Versionen | Sekunden | Tabelle; `OK` bei allen Pins. `ABWEICHUNG`/`FEHLT` ist **Information**, kein Fehler: Die Pins stammen aus dem Audit, nicht aus einem `pip freeze` |
| 5 Smoke-Training `sanity.json`, 2 Mio. Steps | 1–3 Minuten (Anlaufphase langsamer) | `Timestep limit of 2000000 reached`; `smoke_metrics.csv` mit Spalten `ep_end_goal`, `ep_end_timeout`, `ep_length_steps`, `Truncated Steps`; genau eine Kopfzeile; Entropie im letzten Fünftel zwischen 4,2 und 4,5 (frisches Netz, vgl. `docs/phases.md` Phase 1: 4,499 bei 0,1 Mio.); keine `nan` |
| 6 Deployment-Smoke + Bestandsaufnahme | 1–2 Minuten | `deploy_smoke.txt`: `Obs-Größe 257 und 90 Aktionen passen`, Latenz p95 deutlich unter 66,7 ms (in der VM: 0,4 ms auf CPU); `policy_parity.txt`: `3 passed`; `run_inspect.json`: neuester Checkpoint 2704829056, `return_std` ≈ 15,12, `header_lines` 5 (alter Stand), `_git` in `config_used.json` erst nach dem nächsten Trainingsstart |
| 7 Zip | Sekunden | `results\local_check_<datum>.zip` |

Bei Fehlern:

* **Build bricht ab** → `results\local_check_<datum>\build.log`. Steht dort `Upstream-Patch
  fehlt`, hat `apply_patches.ps1` nicht gegriffen: `powershell -File tools\apply_patches.ps1
  -Check` zeigt den Zustand; `git -C third_party\RLGymPPO_CPP status` zeigt fremde Änderungen.
  Steht dort etwas zu `enable_language(CUDA)`, fehlt der libtorch-Patch
  (`tools\patch_libtorch_cuda.ps1`).
* **`Obs-Layout hat sich geaendert`** → **nicht** `update_golden.ps1` ausführen, sondern
  `tests.log` und den frischen Dump (`%TEMP%\rlbot_obs_check_1.json`) zurückgeben. Das Layout
  darf sich nicht geändert haben (Checkpoint-Kompatibilität).
* **Python-Tests < 97 passed** → fehlen `rlbot`/`rlbot_flatbuffers`/`rlgym` im `.venv`?
  `python_versions.txt` zeigt es.
* **Smoke-Training bricht ab** → `smoke_train.log`. `nan` in der ersten Iteration wäre ein
  Build-Problem (libtorch-Version prüfen).
* Jeder Schritt läuft unabhängig weiter; `SUMMARY.md` im Ergebnisordner zeigt alle Stati.

## 2. Nullmessung der Ladder (Stufe 1, Roadmap Punkt 6; ~10–30 Minuten)

Erst jetzt ist die Ladder aussagekräftig (K2 behoben). Über die vorhandenen Checkpoints des
Hauptlaufs:

```powershell
.\.venv\Scripts\python eval\ladder.py --run runs\lucy_1v1 --games 100 --opponents 9
```

Ergebnis: `runs\lucy_1v1\ratings.json` (neuer Schlüssel `lucy_1v1/<steps>`). Erwartung: Die
neueren Checkpoints liegen vorn; ob das Rating mit den Steps monoton steigt, ist genau die
offene Frage aus AUDIT.md K3 (der Skill-Tracker im Training kann konstruktionsbedingt nicht
plateauen). `ratings.json` mit zurückgeben.

## 3. Kontrolllauf Baseline (Stufe 3; ~30–40 Minuten)

```powershell
# Start-Checkpoint: der neueste des Hauptlaufs (Stand Review-Fix: 3907335040), für ALLE
# Experimente derselbe; run_experiment kopiert ihn nach runs\exp_*\start\ und \checkpoints\
$C = (Get-ChildItem runs\lucy_1v1\checkpoints -Directory | Sort-Object { [long]$_.Name } | Select-Object -Last 1).FullName
powershell -ExecutionPolicy Bypass -File tools\experiments\run_experiment.ps1 `
    -Config train\configs\experiments\baseline.json `
    -StartCheckpoint $C -Steps 100000000 -Seed 123
```

Erfolg: `results\exp_baseline_<datum>.zip`; `summary.md` ohne `ABGEBROCHEN`; im letzten Fünftel
etwa: Entropie ≈ 3,58, KL ≈ 0,003, Clip-Fraction ≈ 2–3 %, `ep_end_goal` 0,1–0,4 (bisher nur
geschätzt), `ep_end_time` deutlich unter dem im Audit geschätzten Drittel (K1a: 900 s),
`Truncated Steps` mindestens ~2.048 pro Iteration (pro Spieler gezählt: 1.024 Spiele × 2 Spieler
Blockgrenzen, dazu 2 je Timeout), `Timeout Truncations` = 2 × Timeouts der Iteration,
`Trunc Bootstrap Reset Share` = 0 (K1b-Korrektur, AUDIT.md §7.2b), `Avg Val Target`
≈ 10, SPS in der Größenordnung der bisherigen ~68.000 (lokal nachmessen). Abweichungen sind
keine Fehler, sondern das Ergebnis.

## 4. Experimente (Stufe 3, je ~30–40 Minuten, nacheinander)

Jeweils mit `-Baseline` auf den Ergebnisordner aus Schritt 3:

```powershell
$B = "results\exp_baseline_<datum>"
$C = "<derselbe Start-Checkpoint wie in Schritt 3>"
powershell -ExecutionPolicy Bypass -File tools\experiments\run_experiment.ps1 -Config train\configs\experiments\h2_ent_coef_0004.json -StartCheckpoint $C -Steps 100000000 -Seed 123 -Baseline $B
powershell -ExecutionPolicy Bypass -File tools\experiments\run_experiment.ps1 -Config train\configs\experiments\h3_no_shuffle.json    -StartCheckpoint $C -Steps 100000000 -Seed 123 -Baseline $B
powershell -ExecutionPolicy Bypass -File tools\experiments\run_experiment.ps1 -Config train\configs\experiments\k3_rewards.json       -StartCheckpoint $C -Steps 100000000 -Seed 123 -Baseline $B
powershell -ExecutionPolicy Bypass -File tools\experiments\run_experiment.ps1 -Config train\configs\experiments\zero_sum.json         -StartCheckpoint $C -Steps 100000000 -Seed 123 -Baseline $B
```

Erwartungen aus AUDIT.md: **H2** Entropie < 3,4, Clip-Fraction > 5 %, KL Richtung 0,006
(Warnung des Runners bei Entropie < 2,5). **H3** keine Verschlechterung, eher schnellere
gegnerbezogene Metriken. **K3** `Avg Val Target` sinkt von ~10 Richtung ~3, Value Loss springt
in den ersten Iterationen (Aufwärmphase des Abbruchkriteriums), `Average Step Reward` fällt von
~0,74 auf ~0,2, `ep_end_goal` sollte steigen. **zero_sum** (früher `team_spirit_01`, Review R10):
Im 1v1 ist `team_spirit` selbst wirkungslos, der Wert > 0 schaltet nur `ZeroSumReward` ein, und
jeder Spieler bekommt `r_i − r_j` (eigenes Shaping minus das des Gegners). Tor/Gegentor sind in der
Config auf 5 halbiert, damit ein Tor nach dem Wrapper wie in der Baseline ±10 wert ist; gemessen
wird also nur der Zero-Sum-Effekt auf das dichte Shaping. `Average Step Reward` und
`Average Episode Reward` sind dabei **konstant 0** (Summe beider Spieler) und sagen nichts; das
Shaping-Niveau steht in `raw_step_reward` (Reward vor dem Wrapper, in der Baseline gleich
`Average Step Reward`). Aussagekräftig sind `ep_end_goal`, Ballkontakt und das Duell.

Abbruch (Exit 3) ist ein Ergebnis, kein Fehler: `summary.md` nennt den Grund.

## 5. Vergleich (Sekunden)

```powershell
.\.venv\Scripts\python tools\experiments\compare.py results\exp_* --out results\compare.md
```

`results\exp_*` löst `compare.py` selbst auf (Windows PowerShell reicht das Muster an
`python.exe` wörtlich weiter); `results\exp_*.zip` und Ordner ohne `summary.json` werden
übersprungen und gemeldet (Review R17).

`compare.py` spielt dabei die **gemeinsame Ladder** (alle Experiment-Enden + Baseline-Start +
Baseline-Ende, jeder gegen jeden, 50 Spiele je Paarung; bei 4 Experimenten 15 Paarungen, einige
Minuten). `results\compare.md` und `results\joint_ladder.json` zurückgeben. Hauptkriterium ist
das Duell gegen das Baseline-Ende (Gewinnrate mit 95-%-KI, Review R12); dazu `ep_end_goal`,
Entropie und die gemeinsame Ladder. Die Entscheidung behalten / verwerfen / nachmessen trage ich
in AUDIT.md ein.

## 6. Optional: Stufe 4, Gradientenschritte 6 / 3 / 2 (~1 Stunde)

Erst wenn Stufe 3 entschieden ist:

```powershell
powershell -ExecutionPolicy Bypass -File tools\experiments\bench_expbuffer.ps1 -StartCheckpoint $C -Steps 20000000 -Repeats 2 -DryRun
powershell -ExecutionPolicy Bypass -File tools\experiments\bench_expbuffer.ps1 -StartCheckpoint $C -Steps 20000000 -Repeats 2
```

`-DryRun` zeigt vorher den Plan: Wiederholung r läuft mit Seed 123 + (r − 1), alle Varianten
einer Wiederholung mit demselben Seed, Duell jeweils gegen die 6-Update-Variante derselben
Wiederholung (Review R13). Ergebnis: `results\bench_expbuffer_<datum>.md` (SPS und Lernkurve je
Variante, Streuung über die Wiederholungen).

## 7. Was du mir zurückgibst

1. `results\local_check_<datum>.zip` (Schritt 1)
2. `runs\lucy_1v1\ratings.json` (Schritt 2)
3. `results\exp_*.zip` (Schritte 3 und 4) und `results\compare.md` (Schritt 5)
4. optional `results\bench_expbuffer_<datum>.md` (Schritt 6)
5. bei Fehlern: den jeweiligen Ergebnisordner komplett (Logs liegen drin)

Ich werte mit `compare.py` gegen die Baseline aus, entscheide je Experiment mit Begründung und
schreibe AUDIT.md/Roadmap mit den echten Zahlen fort.

## Hinweise

* `game_timeout_secs` steht in `lucy_1v1.json` jetzt auf 900 (K1a). Ein Fortsetzen des
  Hauptlaufs mit dieser Config ist Checkpoint-kompatibel; der Timeout-Anteil wird ab dann in
  `metrics.csv` als `ep_end_time` sichtbar.
* Der K1-Patch ändert die Value-Targets; der Critic passt sich an, die Policy bleibt gültig
  (AUDIT.md, Checkpoint-Kompatibilität). Alle Experimente inklusive Baseline laufen mit Patch.
* Alte Build-Verzeichnisse (`build\cpp_cu128_avx512`) und Logs (`build_avx512.log`,
  `build_t.log`) sind laut N6 überflüssig; gelöscht wird nichts automatisch.
