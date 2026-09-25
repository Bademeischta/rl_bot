# RLbot: Rocket-League-RL-Bot (Ryzen 7 8700F + RTX 5070)

Umsetzung von „Bauplan v2". Training in C++ (RLGymPPO_CPP + RocketSim, GPU-Learner),
Deployment in Python über RLBot v5.

- Messwerte und Framework-Entscheidung: [docs/phase0_results.md](docs/phase0_results.md)
- Stand der Phasen 1–6: [docs/phases.md](docs/phases.md)

## Struktur
| Ordner | Inhalt |
|---|---|
| `env/cpp/` | Rewards (KRC, Lucy-nah), Obs-Builder mit Padding und Aktions-Stack, State-Setter |
| `env/obs_python.py` | Python-Nachbau des Obs-Builders für das Deployment |
| `train/cpp/` | Trainer, JSON-Config, Env-Factory mit Modus-Mix |
| `train/configs/` | `sanity.json`, `lucy_1v1.json`, `lucy_multimode.json` |
| `eval/` | `duel.cpp` (zwei Checkpoints gegeneinander), `ladder.py` (TrueSkill) |
| `deploy/` | RLBot-v5-Bot, Aktionstabelle, Paket-Adapter, Policy-Laden |
| `tests/` | 75 C++-Tests, 97 Python-Tests, Golden-Fixtures |
| `bench/` | Phase-0-Benchmarks (Python und C++) |
| `tools/` | Audit, Metriken-Anzeige, Policy-Export, Testlauf, Patches (`apply_patches.ps1`), Golden-Prüfung |
| `tools/experiments/` | Stufe-3/4-Experimente: `run_experiment.ps1`, `compare.py`, Abbruchkriterien |
| `tools/local/` | lokales Prüfpaket `run_all_checks.ps1`, Versionsabgleich, Deployment-Smoke |
| `train/configs/experiments/` | je eine Config pro Experiment (ändert genau eine Sache gegenüber `baseline.json`) |
| `third_party/` | gepinnte Klone, libtorch und Upstream-Patches, siehe `PINNED.md` |

## Einrichtung
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install git+https://github.com/AechPro/rlgym-ppo
Copy-Item .venv\Lib\site-packages\rlgym\rocket_league\sim\collision_meshes collision_meshes -Recurse
.\.venv\Scripts\python tools\audit.py
.\.venv\Scripts\python tools\local\check_python_versions.py   # installierte Versionen gegen die Pins
```

`requirements.txt` ist auf den Stand vom 24.09.2026 gepinnt (Audit M2); `rlbot` und
`rlbot_flatbuffers` für Phase 6 sind enthalten.

libtorch (CPU und cu128) nach `third_party/libtorch_cpu` bzw. `third_party/libtorch_cu128`
entpacken (URLs in `third_party/PINNED.md`), dann:

```powershell
powershell -ExecutionPolicy Bypass -File tools\patch_libtorch_cuda.ps1
powershell -ExecutionPolicy Bypass -File bench\cpp\build.ps1 -Flavor cu128
```

## Training
```powershell
.\build\cpp_cu128\train_bot.exe train\configs\lucy_1v1.json
.\.venv\Scripts\python tools\show_metrics.py runs\lucy_1v1\metrics.csv
```
Derselbe Befehl setzt einen abgebrochenen Lauf am jüngsten Checkpoint fort.

## Bewerten
```powershell
.\.venv\Scripts\python eval\ladder.py --run runs\lucy_1v1 --games 100
```

## Deployment
```powershell
.\.venv\Scripts\python tools\export_policy.py runs\lucy_1v1\checkpoints --out deploy\rlbot\policy.pt
```
Dann `deploy/rlbot/bot.toml` im RLBot-Launcher laden.

Der Bot entscheidet wie im Training auf dem Paket von vor 7 Ticks (Audit H1). Rückweg zum
Verhalten davor (aktuelles Paket): vor dem Start von RLBot in derselben Shell
`$env:RLBOT_OBS_DELAY = "0"` setzen (erlaubt 0 bis 7, Default 7).

## Tests
```powershell
powershell -ExecutionPolicy Bypass -File tools\run_all_tests.ps1 -Repeat 2
```

Die Golden-Fixtures (`tests/fixtures/obs_golden.json`) werden dabei **nicht** überschrieben,
sondern numerisch gegen einen frischen Dump geprüft (`tools/check_golden.py`). Weicht der Dump
ab, hat sich das Obs-Layout geändert und alle Checkpoints wären inkompatibel; bewusst
aktualisieren nur mit `tools\update_golden.ps1`.

## Audit und Roadmap
Befunde, Roadmap und Herkunft aller Messwerte: [AUDIT.md](AUDIT.md). Umsetzungsstand:
[AUDIT_PROGRESS.md](AUDIT_PROGRESS.md). Was lokal auszuführen ist: [LOCAL_RUNBOOK.md](LOCAL_RUNBOOK.md).

## Phase-0-Ergebnis (23.09.2026)
**RLGymPPO_CPP mit cu128-libtorch auf der GPU**, `numThreads=16`, `numGamesPerThread=64`:
**117.247 Overall SPS** gegen 18.754 SPS bei der besten Python-Konfiguration (6,3×).
Die RTX 5070 (sm_120) funktioniert; nötig war nur ein CMake-Patch an libtorch
(`tools\patch_libtorch_cuda.ps1`), weil nvcc 12.8 den MSVC von VS 2026 ablehnt.

## Regeln
- Deployment nur offline über „Launch without EAC", nie online oder ranked.
- Nach Rocket-League-Patches die Collision-Meshes prüfen (`tools/DUMP_MESHES.md`).
