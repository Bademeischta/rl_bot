# AUDIT_PROGRESS: Umsetzungsstand der Audit-Roadmap

Zweck: Eine neue Session (Mensch oder Agent) soll hier ohne weiteres Vorwissen einsteigen können.
Grundlage ist `AUDIT.md` (Befunde K1–N10, Roadmap Stufen 0–4). Dieses Dokument wird bei jedem
Roadmap-Punkt fortgeschrieben.

Branches: `claude/rlbot-audit-roadmap-c8t7nl` (Audit-Roadmap, abgezweigt von `main` @ `54105bf`,
als PR #1 in `main` gemergt) und `claude/review-fixes` (Review-Befunde R1-R19, von `main` @
`bb7f93e`, eigener PR). Regel: ein Commit pro Punkt, ID in der Commit-Message.

## Stufe 3: Duell-Instrument und Experimente (Branch `claude/duel-and-experiments`, ab 26.09.2026)

Abgezweigt von `claude/review-fixes` (8c6e7fa); dieser Branch war entgegen der Annahme im Auftrag
noch nicht in `main` gemergt (Nutzer-Entscheidung: von review-fixes abzweigen). Regeln: Hauptlauf
wird nicht fortgesetzt, Experimente nacheinander, alle vom neuesten Checkpoint (3.907.335.040),
Seed 123, derselbe Build (Git-Hash in `config_used.json`).

### Schritt 1: Duell-Auswertung (erledigt)

| ID | Änderung | Test (ohne Fix rot) |
|---|---|---|
| D1 | `duel.exe`: 300-s-Matches mit Anstoß nach jedem Tor, Ergebnis je Spiel, geseedete Anstöße (Paare mit Seitentausch), je Spiel frische Arena + eigene Generatoren (Aktionen, RocketSim-Respawn), Car-ID-Reihenfolge, Sperre für deterministische Wiederholungen, parallele Spiele (`--threads`) | `tests/test_duel.py` (echter `duel.exe`, 4 rot mit dem alten) |
| D2 | Hauptkriterium in `compare.py`/`summarize.py`: Tordifferenz pro Spiel mit 95-%-t-KI; Gewinnrate (Wilson) und Tore/min zusätzlich | `test_experiments_tools.py` (3 rot mit dem alten) |
| D3 | Spielanzahl 1000 je Duell (`run_experiment.ps1`, `bench_expbuffer.ps1`), gemeinsame Ladder 100 Spiele je Paarung | `test_default_duel_games_resolve_the_target_effect` (rot mit 100) |

Prüfungen zur Unabhängigkeit: verschiedene Anstoß-Seeds je Spielpaar (ja), Seitentausch (ja, Blau
gegen Orange in der Nullmessung +0,021 [−0,032; +0,074]), deterministische Policies (ja, 1 von 20
Spielen exakt doppelt → Sperre), stochastisch 1000/1000 verschiedene Spiele. Nullmessung,
Laufzeit, Effekt von 100/225 Mio. Steps und die Begründung für 1000 Spiele: AUDIT.md §7.8.
Rohdaten: `results\duel_null\*.json`.

## Review-Fixes (Branch `claude/review-fixes`, lokal auf dem Trainings-PC, ab 25.09.2026)

Ein unabhängiger Review hat nach dem Merge von PR #1 Fehler gefunden. Behoben wird lokal auf dem
Windows-PC (Ryzen 7 8700F, RTX 5070, VS 2026, Windows PowerShell 5.1), ein Commit pro Befund
(`R<n>` in der Commit-Message). Der Review-Bericht selbst lag nicht vor; Grundlage ist die
Befundliste aus dem Auftrag.

| ID | Befund | Status | Test (schlägt ohne Fix fehl) |
|---|---|---|---|
| R1 | Patches mit CRLF (autocrlf) → `git apply` scheitert am gepinnten Commit | behoben | `tests/test_build_scripts.py` (LF im Arbeitsverzeichnis, `check-attr`, frischer Checkout + `git apply`) |
| R2 | PowerShell 5.1: stderr nativer Befehle wird mit `ErrorActionPreference=Stop` zum Abbruch | behoben: `tools/NativeCommand.ps1` (`Invoke-Native`), alle 8 Skripte umgestellt (44 Aufrufe) | `test_invoke_native_*` (stderr + Exit-Code unter 5.1), `test_apply_patches_ps1_runs_under_ps51_on_a_fresh_checkout` (echtes Skript, frischer Klon), AST-Lint `tests/ps_lint_native_calls.ps1` |
| R3 | `.ps1` ohne BOM: Umlaute/typografische Zeichen brechen unter 5.1 die Syntax | behoben: alle `.ps1` mit Nicht-ASCII-Zeichen als UTF-8 mit BOM (9 Dateien), Lint-Skript reines ASCII | `test_every_ps1_is_ascii_or_utf8_with_bom`, `test_every_ps1_parses_and_reads_identically_under_ps51` (liest wie `powershell -File`, parst, vergleicht mit UTF-8) |
| R4 | K1b: Timeout-Bootstrap mit der Reset-Obs statt der letzten Obs der Episode | behoben: `StepResult::finalObs` (GameInst sichert vor dem Reset, `std::move`), ThreadAgent bootstrappt davon; Diagnose `Timeout Truncations`, `Trunc Bootstrap Reset Share` (muss 0 sein), `Trunc Bootstrap V Final/Reset/Diff`; Schalter `env.timeouts_as_truncation` (Default an); CMake verlangt `RLGSC_HAS_FINAL_OBS`; `apply_patches.ps1 -Reset`; AUDIT.md §7.2/7.2a korrigiert, §7.2b neu (Truncated Steps pro Spieler, im 1v1 ≥ ~2.048) | `tests/cpp/test_truncation.cpp`: `K1b_Echter_Pfad_…` (Gym → GameInst → ThreadAgent → Learner/GAE; mit dem alten `stepResult.obs` lokal nachweislich rot), `K1b_GameInst_…`, Schalter-Tests; `test_apply_patches_reset_…` |
| R5 | NaN wird als leeres Feld geschrieben, `check_abort` ignoriert leere Felder | behoben: Writer schreibt `nan`/`inf`/`-inf` wörtlich; `metrics_util`/`check_abort` werten nan, inf **und** leere Felder als Abbruch (Entropie, Value Loss, KL, Advantage, Val Target); `Average Episode Reward` nach Nutzerentscheidung ausgenommen und in `summary.md` gezählt; halb geschriebene letzte Zeile wird ignoriert | `tools/cpp/write_metrics_csv.cpp` (echter `MetricsCSVWriter`) → `check_abort.py`: nan/inf/-inf/fehlender Schlüssel lösen aus; C++ `CSV_nan_und_inf_werden_woertlich_geschrieben`; ohne Fix 13 Python-Tests rot (geprüft) |
| R6 | `env.seed_envs` Default true ändert das Verhalten bestehender Configs | behoben: Default false (altes Verhalten, `lucy_1v1.json` unverändert), alle 8 Experiment-Configs setzen `"seed_envs": true` ausdrücklich | C++ `Config_shuffle_slots_und_seed_envs_haben_altes_Verhalten_als_Default`; `test_every_experiment_config_seeds_its_envs_explicitly`, Baseline-Diff erlaubt nur `env.seed_envs`; ohne Fix rot (geprüft) |
| R7 | `config_used.json` (`_git`, `_started`) ist nicht mehr als Config ladbar | behoben: Felder mit führendem `_` werden auf jeder Ebene ignoriert, andere unbekannte weiter abgelehnt; `TrainConfig::ToUsedJSONString` erzeugt `config_used.json` (von `main.cpp` benutzt) | `Config_config_used_json_ist_wieder_als_Config_ladbar` (echter Inhalt von `config_used.json`), `Config_Unterstrich_Felder_…`; mit altem Parser (nur `_comment`) rot (geprüft) |
| R8 | H1-Paketpuffer ohne Rückweg | behoben: Umgebungsvariable `RLBOT_OBS_DELAY` (0–7, Default 7 wie im Training); 0 = exakt das Verhalten vor H1 (aktuelles Paket, keine Phasen-Logik); dokumentiert in `bot.py` und README | `test_obs_delay_default_is_7_and_env_var_overrides`, `…_sets_the_buffer_length` (Delay 3), `…_zero_is_exactly_the_behaviour_before_h1`; mit altem `bot.py` rot (Import) |
| R9 | Alte Rating-Schlüssel (nur Step-Zahl) nicht mehr lesbar | behoben in `eval/ladder.py`: alte Schlüssel werden beim Laden mit dem Laufnamen ergänzt (`--run-name`, sonst Ordnername), `--show` zeigt jede Datei read-only; die Ladder schreibt nie in eine Datei mit alten Schlüsseln (Abbruch mit Hinweis), `--migrate … --out <kopie>` schreibt nur eine Kopie. `deploy/watch.py` liest keine Ratings (nur Checkpoints, M5) – dort kein Handlungsbedarf | `test_legacy_keys_stay_readable_…`, `test_migration_only_writes_a_copy`, `test_ladder_cli_refuses_…` (Datei byte-gleich), echte `runs/sanity/ratings.json` read-only; mit altem `ladder.py` 4 rot |
| R10 | `team_spirit_01` ist im 1v1 wirkungslos (τ), misst in Wahrheit Zero-Sum mit doppeltem Torwert | behoben: umbenannt in `zero_sum.json`, ehrlich beschrieben (τ wirkungslos, `r_i − r_j`), `goal`/`concede` 5 → Tor nach dem Wrapper ±10; `RawRewardTap` vor dem Wrapper, Metrik `raw_step_reward` (auch in Zusammenfassung/Vergleich); Runbook und README korrigiert | C++ `ZeroSum_1v1_Tor_bleibt_bei_10_…` (echte Arena, Tor ins blaue Tor), `ZeroSum_1v1_team_spirit_ist_wirkungslos` (τ 0,1/0,5/1,0 identisch, Summe 0), `ZeroSum_roher_Reward_…`; Config-Test erwartet genau team_spirit + goal + concede |
| R11 | K3 ist ein Bündel aus 7 Werten, nicht als solches markiert | behoben: `compare.py` liest die Änderungen aus den `config.json` der Läufe und markiert jedes Experiment mit > 1 Änderung als Bündel (Tabelle, Änderungs-Abschnitt, Hinweis „nur bei schlechtem/unklarem Ergebnis aufteilen“); K3-Config-Kommentar, experiments/README und AUDIT.md §7.6 nennen das Bündel | `test_compare_marks_k3_as_a_bundle_of_seven_values` (echte Configs baseline/k3/h2/zero_sum); mit altem `compare.py` rot |
| R12 | `compare.py`: TrueSkill aus getrennten Ladders nicht vergleichbar; Duell ohne Konfidenzintervall | behoben: `compare.py` spielt EINE gemeinsame Ladder (Baseline-Start, Baseline-Ende, alle Experiment-Enden, jeder gegen jeden, `--ladder-games`, Ergebnis `joint_ladder.json`), Einzel-Ladders werden ignoriert; Hauptkriterium = Duell gegen Baseline-Ende, Gewinnrate (Remis halb) mit 95-%-Wilson-KI in Tabelle, Hinweisen und `summary.md` | `test_compare_plays_one_joint_ladder_…` (echtes `duel.exe` auf echten Checkpoints, nur gelesen), `test_win_rate_ci_matches_the_wilson_interval` (Referenzwerte), Tabelle/Hinweise/summarize; mit altem Code 5 rot |
| R13 | `bench_expbuffer.ps1`: gleicher Seed in allen Wiederholungen | behoben: Wiederholung r bekommt `-Seed + (r − 1)`, alle Varianten einer Wiederholung denselben (gepaart); Duell-Baseline ist die 6-Update-Variante derselben Wiederholung; `-DryRun` zeigt den Plan; `-LadderGames` für die gemeinsame Ladder | `test_bench_expbuffer_uses_a_different_seed_per_repetition` (echter Plan des Skripts unter 5.1); mit altem Skript rot |
| R14 | `run_experiment.ps1`: Start-Checkpoint liegt in der Checkpoint-Rotation | behoben: zwei per Hash geprüfte Kopien, `runs\exp_…\start\<steps>` (Referenz fürs Duell „Ende gegen Start“ und `start_checkpoint` in `summary.json`, außerhalb der Rotation) und `checkpoints\<steps>` (lädt der Trainer); Zip wird nie überschrieben; `-PrepareOnly`, `-RunsRoot`, `-ResultsRoot` für Tests | `test_run_experiment_keeps_the_start_checkpoint_outside_the_rotation` (echtes Skript, echter neuester Checkpoint nur gelesen, Hashes); mit altem Skript rot |
| R15 | `run_experiment.ps1`: Trainer wird hart beendet (`Stop-Process -Force`) | behoben: `train_bot.exe --stop-file` (Training endet nach der laufenden Iteration, `save_on_exit` schreibt den End-Checkpoint; alte Stop-Datei beim Start = Fehler); `Stop-TrainerGracefully` (`tools/experiments/TrainerControl.ps1`) legt die Datei an und beendet nur nach `-StopTimeoutSeconds` (Default 600) als Notfall hart; ExitCode-Fix für 5.1 (`$proc.Handle`) | `test_train_bot_stops_cleanly_after_the_iteration_and_saves` (echter Trainer, CPU, vollständiger End-Checkpoint), `…_refuses_a_stale_stop_file`, `test_stop_trainer_gracefully_…` (reagierender und hängender Prozess); ohne Fix 3 rot |
| R16 | `run_experiment.ps1`: End-Checkpoint ohne Vollständigkeitsprüfung | behoben: `tools/experiments/pick_checkpoint.py` wählt den neuesten vollständigen Checkpoint (alle 5 Dateien nicht leer, 4 `.lt` intakte Zip-Archive mit CRC, `RUNNING_STATS.json` passt zum Ordnernamen, Ladeprüfung mit `deploy.policy.load_policy`) und meldet verworfene mit Grund; `run_experiment.ps1` benutzt ihn | `test_pick_checkpoint_skips_incomplete_newer_checkpoints` (echte Checkpoint-Dateien als Kopie, halb geschriebener Optimizer, fehlende Stats, falsche Steps, leere Policy), `…_fails_without_any_complete_checkpoint`; ohne Werkzeug rot |
| R17 | Runbook-Vergleichsbefehl verlässt sich auf Glob-Expansion der Shell | behoben: `compare.py` löst Muster selbst auf (`glob`), nimmt nur Ordner mit `summary.json`, meldet übersprungene (`exp_*.zip`, halbe Ordner), Muster ohne Treffer = Fehler; Runbook/README erklären es | `test_compare_expands_the_glob_pattern_itself_like_under_powershell` (Argumentliste ohne Shell wie unter PowerShell, mit Zip und halbem Ordner); mit altem `compare.py` rot |
| R18 | `run_all_checks.ps1`: harte Branch-Prüfung, Folgeschritte auf alten Binaries, Ergebnisordner überschreibbar | behoben: Schritt 1 prüft „Arbeitsverzeichnis sauber“ und gibt Branch, vollen Hash und Upstream-Hash aus; Schritte deklarieren Abhängigkeiten (3, 5, 6 brauchen den Build, 2 braucht 1) und werden sonst „übersprungen wegen Schritt n“; `-SkipBuild` verlangt Binaries jünger als HEAD; Ergebnisordner `local_check_<datum>_<hhmmss>`, Ordner/Zip nie überschrieben; Exit 1 bei Fehler/Übersprung; `results/` in `.gitignore`; Smoke mit absoluten Pfaden | `test_run_all_checks_skips_dependent_steps_…` (echtes Skript unter 5.1 mit fehlenden Binaries), `test_results_folder_is_ignored_by_git`; mit altem Skript rot |
| R19 | Nebenbefund (lokal, mit OK des Nutzers): geseedete Szenen verteilten ihre Zufallszahlen in Hash-Reihenfolge von `arena->_cars` (`unordered_set`, adressabhängig) auf die Autos; gleicher Seed ≠ gleiche Startzustände, `EnvFactory_reicht_Seed_…` flackerte (1 von 5) | behoben: Szenen iterieren nach Car-ID, Dribble-Ballführer zieht der geseedete RNG (weiter gleichverteilt); Test ordnet Spieler nach Car-ID zu | `Seed_gleicher_Seed_gibt_gleiche_Autozustaende_unabhaengig_von_der_Speicherreihenfolge` (40 Arenen, beide Reihenfolgen); beide Tests ohne Fix rot, mit Fix 3× grün |

Nachträge im selben Befund (eigene Commits): „R2 Nachtrag" (leere stderr-Zeilen),
„R4 Nachtrag" (`tools/local/check_k1b.py` prüft die K1b-Diagnose im Smoke-Schritt),
„R10 Nachtrag" (ZeroSum-Test ordnete nach Index statt Car-ID und flackerte, wie R19),
„R18 Nachtrag" (Zip mit Wiederholung, falls eine Datei kurz gesperrt ist).

Nebenbefund bei der Bestandsaufnahme: Der Hauptlauf `runs/lucy_1v1` wurde nach dem Audit mit dem
**alten** Binary (ohne K1b, `game_timeout_secs` 300) bis 3.907.335.040 Steps weitertrainiert; der
Checkpoint `2704829056` aus Runbook und Audit existiert wegen `checkpoints_to_keep = 10` nicht
mehr (Runbook nimmt jetzt immer den neuesten). Die alten Binaries sind vor dem Neubau nach
`build/cpp_cu128_vor_review_2026-09-25/` gesichert. `runs/lucy_1v1` wurde nur gelesen.

### Verifikation lokal (25.09.2026, Windows PowerShell 5.1, MSVC 14.51, cu128)

| Prüfung | Ergebnis |
|---|---|
| `git apply --check` beider Patches auf frischem Checkout von `ee4cc56` | OK (Test `test_patches_apply_to_a_fresh_checkout_of_the_pinned_commit`, dazu `apply_patches.ps1` unter 5.1) |
| Erster MSVC-cu128-Build mit dem Truncation-Patch (alte Fassung, vor R4) | fehlerfrei, 75/75 C++-Tests; danach alle Builds mit der neuen Fassung fehlerfrei |
| Alle `.ps1` unter `powershell.exe` 5.1 geparst und wie `-File` gelesen | 11 Skripte, 0 Parserfehler, textgleich zu UTF-8 |
| `tools\local\run_all_checks.ps1` unter 5.1, Commit `9d62f94`, sauberes Arbeitsverzeichnis | **alle 6 Schritte OK, Exit 0, 2,4 min** (`results\local_check_2026-09-25_191736.zip`) |
| C++-Tests (`rlbot_tests`), 2 Durchläufe im Prüfpaket + 12 Durchläufe einzeln | **88/88** in jedem Durchlauf |
| Python-Tests, 2 Durchläufe | **147 passed, 0 skipped** je Durchlauf (Paritätstests gegen den echten Checkpoint) |
| Golden-Fixtures | unverändert (120 Obs, Aktionstabelle, 47 Rotationen, 1e-5) |
| Smoke `sanity.json`, 2 Mio. Steps, GPU | 20 Iterationen, eine Kopfzeile, kein `nan`, Entropie ~4,48, ~74.000 SPS; **K1b-Diagnose OK**: 7.654 Timeout-Truncations in 17 Iterationen, `Trunc Bootstrap Reset Share` überall 0, V(letzte Obs) − V(Reset-Obs) im Mittel −0,74; `Truncated Steps` 2.180 = ~2.046 Blockgrenzen + 134 Timeouts; `_git` = `9d62f94` (nicht dirty); `raw_step_reward` = `Average Step Reward` (kein Wrapper) |
| Deployment-Smoke mit dem echten Checkpoint 3907335040 | Obs 257 / 90 Aktionen passen, Latenz p95 0,28 ms (Budget 66,7 ms), Policy-Parität 3/3 |
| `run_experiment.ps1` Mini-Modus, `baseline.json`, 3 Mio. Steps, 20 Duell-/10 Ladder-Spiele | **komplett durch, Exit 0, 1,6 min**: `start\` + `checkpoints\` per Hash geprüft, Training ~71.000 SPS, End-Checkpoint 3910425600 (`save_on_exit`, `pick_checkpoint.py`), Duell Ende gegen Start (Gewinnrate 47,5 %, KI [27,9 %, 68,0 %], 17 von 20 remis), Lauf-Ladder, `summary.md/json`, Zip |
| `compare.py "results/exp_mini_*"` mit gemeinsamer Ladder | Muster selbst aufgelöst (Zip übersprungen), Ladder Baseline-Start gegen Baseline-Ende mit `duel.exe` (10 Spiele, 9 remis), `results\mini_compare.md`, `results\joint_ladder.json` |

### Was jetzt noch ungetestet ist

* Der **Abbruchpfad** von `run_experiment.ps1` am Stück (check_abort löst aus → Stop-Datei →
  Trainer beendet sauber → Zusammenfassung mit `ABGEBROCHEN`): Die Bausteine sind einzeln getestet
  (`--stop-file` am echten Trainer, `Stop-TrainerGracefully` mit reagierendem und hängendem
  Prozess, `check_abort` mit echter CSV), der Gesamtablauf lief nicht, weil im Mini-Lauf nichts
  abbrach.
* Das **Duell gegen das Baseline-Ende** in `run_experiment.ps1` (`-Baseline`) und damit die
  Hauptkriteriums-Spalte mit echten Daten: nur mit Testdaten geprüft (es gab bewusst nur einen
  Mini-Lauf, kein zweites Experiment).
* `bench_expbuffer.ps1` außerhalb von `-DryRun` (würde sechs Trainingsläufe starten).
* `tools\update_golden.ps1` (bewusst nicht ausgeführt: würde die Referenz überschreiben).
* Wirkung von K1b, K1a, H6/R19 auf das Lernen; alle Experimente (Stufe 3/4) – laut Auftrag nicht
  gestartet.
* H1/R8 im echten Spiel über RLBot (Paketfolge, Phasenwechsel, `RLBOT_OBS_DELAY`).
* `config_used.json` eines echten Laufs als Config für `train_bot.exe` (nur per Unit-Test mit dem
  echten Erzeugungscode geprüft).

### Offen / zu entscheiden

* **Duell-Remis:** Im Mini-Lauf endeten 17 von 20 Duellspielen und 9 von 10 Ladder-Spielen ohne
  Tor (`duel.exe` bricht nach 120 s Spielzeit ab). Das Hauptkriterium „Gewinnrate mit 95-%-KI"
  wird mit 100 Spielen deshalb breite Intervalle haben. Vor den Experimenten entscheiden: mehr
  Spiele (`-DuelGames`), längere Spiele (`--max-seconds`) oder zusätzlich den Toranteil gewichten.
* Kosmetik unter 5.1: Umlaute in Ausgaben nativer Programme (git-Commitbetreff in `git.txt`,
  Python-Meldungen) erscheinen in den Logs verstümmelt, weil PowerShell deren Ausgabe mit der
  OEM-Codepage liest. Inhaltlich ohne Folgen; nicht geändert, weil MSVC-Meldungen umgekehrt in
  dieser Codepage kommen.
* Beim Verifizieren angelegt (nicht gelöscht, alles neu und außerhalb von `runs\lucy_1v1`):
  `runs\local_check_2026-09-25_161247|191339|191736\` (Smoke-Läufe), `runs\exp_mini_baseline_2026-09-25_192023\`
  (Mini-Lauf), `runs\local_check_2026-09-25_1611\` (leer, Gegenprobe mit dem alten Skript) und die
  zugehörigen Ordner/Zips in `results\` (ignoriert). Können gelöscht werden.

## Arbeitsumgebung der Umsetzung (25.09.2026)

Cloud-Session auf einer Linux-VM (4 Kerne, 15 GB RAM, keine GPU, kein Windows, kein Rocket League,
kein `runs/`-Ordner mit echten Checkpoints). Alles, was Hardware braucht, ist als lokales Paket
vorbereitet (`LOCAL_RUNBOOK.md`, `tools/local/`, `tools/experiments/`).

Was in der VM geprüft wurde (Stand nach dem letzten Commit):

| Prüfung | Ergebnis |
|---|---|
| Python-Tests (`.venv`: numpy 1.26.4, torch 2.14.0+cpu, rlgym 2.0.1, rlbot 2.0.0b55, rlbot_flatbuffers 0.19.0, trueskill 0.4.5) | **97 bestanden, 3 übersprungen** (Policy-Parität braucht `runs/lucy_1v1`); mit `RLBOT_PARITY_RUN` auf einen VM-Smoke-Checkpoint auch diese 3 grün |
| C++-Build unter Linux (GCC 13, CPU-libtorch aus dem pip-Wheel, Python-3.11-Header) mit beiden Upstream-Patches | alle Targets gebaut (`train_bot`, `rlbot_tests`, `dump_obs`, `dump_policy_actions`, `duel`, `bench_cpp_sps`) |
| C++-Tests (`rlbot_tests collision_meshes`, Meshes aus dem `rlgym`-Wheel) | **75 bestanden, 0 fehlgeschlagen** (vor dem Audit 39) |
| Upstream-Patches | `git apply --check` auf dem reinen `ee4cc56` und nach dem GCC-Patch; Idempotenz-Erkennung (`--reverse --check`) |
| Golden-Fixtures | Linux-`dump_obs` gegen die Windows-Referenz: 120 Obs-Vektoren, größte Abweichung 9,5e-7 (deshalb prüft M3 numerisch, nicht per Hash) |
| Smoke-Trainingsläufe `train_bot` auf CPU (2×2 Spiele, Netz 32×32, wenige tausend Steps) | Neustart aus Checkpoint, `metrics.csv` mit einer Kopfzeile, `ep_end_*`/`scene_*`/`Truncated Steps`, `extra_steps`, `save_on_exit`, `_git` in `config_used.json` |
| `duel` + `eval/ladder.py`, `tools/local/deploy_smoke.py`, `tools/local/inspect_run.py`, `tools/check_golden.py`, `tools/local/check_python_versions.py` | auf den VM-Smoke-Checkpoints ausgeführt |
| PowerShell-Skripte (alle 8 `*.ps1`) | mit PowerShell 7.4 (Linux) nur **geparst**: keine Syntaxfehler; **nie ausgeführt** (Windows-Pfade, `.exe`) |

**Nicht in der VM prüfbar** (alles im `LOCAL_RUNBOOK.md`): MSVC-Build mit cu128, die Patches unter
MSVC, Laufzeit der Skripte, jede Zahl zu SPS/Dauer, Ladder-Nullmessung, Experimente, echte
Deployment-Latenz, RLBot-Spiel. **Keine Performance-Zahl aus dieser VM wird als Messwert verwendet.**

## Statustabelle

Status-Werte: `offen` · `umgesetzt (VM-getestet)` · `umgesetzt (ungetestet, lokal prüfen)` ·
`vorbereitet (lokal ausführen)` · `verworfen` · `außerhalb Scope`.

| ID | Stufe | Inhalt | Status | Commit | Test / Nachweis |
|---|---|---|---|---|---|
| H4 | 0 | Git-Commit + Tag, Git-Hash in `config_used.json` | Commit `54105bf` existiert (vom Nutzer, nach dem Audit); `config_used.json` trägt jetzt `_git` (Build-Hash, `-dirty`-Suffix) und `_started`; **Tag `baseline-2.7G` lokal setzen** (Runbook Schritt 0) | Schritt 2a | VM-Smoke-Lauf: `_git` = Build-Hash |
| M8 | 1 | Metriken `ep_end_goal`, `ep_end_timeout`, `ep_end_notouch`, `ep_end_time`, `ep_end_truncated`, `ep_length_steps`, `scene_<name>_goal/_length` | umgesetzt (VM-getestet) | M8 | `tests/cpp/test_metrics.cpp` (8); `OnIteration` aggregiert alle `AccumAvg`-Schlüssel dynamisch |
| H6 | 1 | Seed an eigene State-Setter und Obs-Shuffle (`env.seed_envs`, Default **false** seit R6 = alter zeitgeseedeter Pfad; Experimente setzen true) | umgesetzt (VM-getestet) | H6 | `Seed_*`, `OBS_Shuffle_mit_Seed_*`, `EnvFactory_reicht_Seed_*`. Nicht seedbar bleiben Upstream-Teile: `RandomState`, `Arena::ResetToRandomKickoff`, SkillTracker-Seitentausch |
| M1 | 1 | `metrics.csv`: Kopfzeile aus Datei übernehmen, neue Spalten anhängen, 12 signifikante Stellen; `nan`/`inf` seit R5 **wörtlich** (N4 hatte leer geschrieben, das übersah `check_abort`) | umgesetzt (lokal getestet) | M1, R5 | `CSV_*`-Tests; Smoke-Lauf mit einer Kopfzeile |
| K2 | 1 | `duel.cpp` nutzt `gym.Reset()`/`result.obs` statt `BuildOBS` doppelt | umgesetzt (VM-getestet) | K2 | Linux-`duel` auf Smoke-Checkpoints; `EnvFactory_Env_laeuft_100_Schritte_*` (Stack wandert um genau eine Aktion). **Ladder-Nullmessung lokal** (Runbook Schritt 2) |
| M4 | 1 | Rating-Schlüssel `<lauf>/<steps>`, `ratings.json` je Lauf | umgesetzt (VM-getestet) | K2 | `test_rating_key_*`; `ladder.py --run` in der VM |
| M5 | 1 | `watch.py --run`, Policy-Paritätstest nur Hauptlauf, numerisch sortiert; `RLBOT_BUILD_DIR`/`RLBOT_PARITY_RUN` | umgesetzt (VM-getestet) | K2 | `test_latest_checkpoint_*`; Paritätstests gegen Linux-Binary grün |
| K1a | 2 | `game_timeout_secs` 900 in `lucy_1v1.json` und `lucy_multimode.json`; `sanity.json` bleibt 120 s (AUDIT.md 7.5) | umgesetzt | K1a | Wirkung lokal über `ep_end_time` (M8) |
| H1 | 2 | `PacketBuffer` in `deploy/rlbot/bot.py`: Entscheidung auf dem Paket von vor 7 Ticks; Replay/Countdown/Pause leeren Puffer und Aktions-Stack (wie ein Reset im Training) | umgesetzt (VM-getestet) | H1 | 10 Tests (`tests/test_bot_logic.py`): Delay 7, erste Entscheidung Delay 0, verpasste Ticks, große Lücke, Tor-Replay/Kickoff, Pause, Duplikate, Rücksprung. Echte Latenz: **lokal** |
| M3 | 2 | Golden-Fixtures werden nur noch geprüft (`tools/check_golden.py`, Toleranz 1e-5), Referenz bewusst per `tools/update_golden.ps1` | umgesetzt (VM-getestet) | M3/M7/M2 | `tests/test_tools_checks.py` |
| M7 | 2 | `check_policy_compatible()` in `bot.py`; N1 toter Code entfernt | umgesetzt (VM-getestet) | M3/M7/M2 | 3 Tests |
| M2 | 2 | `requirements.txt` gepinnt (inkl. `rlbot`, `rlbot_flatbuffers`); `tools/local/check_python_versions.py`; `run_all_tests.ps1` verlangt ≥ 60 ausgeführte Python-Tests | umgesetzt; **Pins ungeprüft gegen den PC** | M3/M7/M2 | Runbook Schritt 1.4 meldet Abweichungen |
| K1b | 2 | Upstream-Patch `rlgympppo_cpp_truncation.patch`: `IsTruncation()`, `StepResult::truncated`, ThreadAgent, letzte Episoden-Obs in `nextStates`, GAE-Bootstrap mit `V(nextStates)`; `apply_patches.ps1` in `build.ps1`; CMake bricht ohne Patch ab | **erste Fassung fehlerhaft** (Bootstrap von der Reset-Obs), korrigiert in R4; MSVC-cu128 lokal gebaut und getestet | K1b, R4 | `K1_*` + `K1b_*` (echter Pfad bis zur GAE); Smoke-Lauf: Reset-Anteil 0 |
| H2 | 3 | `ent_coef` 0,004 — Experiment-Config `h2_ent_coef_0004.json` | vorbereitet (lokal ausführen) | Schritt 2b | `tests/test_experiment_configs.py`: genau eine Änderung |
| H3 | 3 | Schalter `env.shuffle_slots` (Default true = bisher); Experiment `h3_no_shuffle.json` | Schalter umgesetzt (VM-getestet), Experiment vorbereitet | H6, 2b | `OBS_Shuffle_Slot0_Anteil_ist_ein_Drittel` bestätigt die Audit-Aussage „ein Drittel" |
| K3 | 3 | Reward-Umgewichtung laut AUDIT.md — Experiment `k3_rewards.json`; `RUNNING_STATS.json` wird übernommen (AUDIT.md 7.3) | vorbereitet (lokal ausführen) | Schritt 2b | Config-Test; Abbruchkriterium mit Aufwärmphase |
| — | 3 | Zero-Sum-Shaping — Experiment `zero_sum.json` (früher `team_spirit_01.json`, siehe R10) | vorbereitet (lokal ausführen) | Schritt 2b, R10 | Config-Test, `ZeroSum_*` |
| — | 3 | Trainer-Optionen `extra_steps`, `save_on_exit`, `--stop-file` (R15); `run_experiment.ps1`, `check_abort.py`, `summarize.py`, `compare.py`, `pick_checkpoint.py` | umgesetzt, lokal getestet (Mini-Lauf komplett, R12-R17) | Schritt 2a/2b, R12-R17 | siehe Review-Tabelle |
| H5 | 4 | `learner.exp_buffer_iterations` (Default 3), Configs 6/3/2 Updates, `bench_expbuffer.ps1` | Option umgesetzt (VM-getestet), Benchmark vorbereitet (lokal) | H5 | Config-Tests; „+10 %" bleibt Schätzung bis zur lokalen Messung |
| M6 | 4 | Obs-Allokationen | offen — H3 wurde als Schalter, nicht als Slot-Permutations-Umbau umgesetzt; Roadmap sieht M6 nur als Nebeneffekt vor | — | — |
| N6 | 4 | AVX-512-Zweig | **verworfen**: lokal auf dem Ryzen gemessen (3 gg. 3, −0,8 % in der Streuung), kein Benchmark-Skript; Ordner/Logs bleiben liegen | — | AUDIT.md 7.4 |
| N5 | — | `PINNED.md` | erledigt mit K1b (Patch-Tabelle, Verweis auf `bench/cpp/CMakeLists.txt` entfernt) | K1b | — |
| N10 | — | Test mit echtem Trainings-Env | erledigt mit H6 (`EnvFactory_Env_laeuft_100_Schritte_*`) | H6 | — |
| M9, N2, N3, N7, N8, N9 | — | nicht in der Roadmap | offen (bewusst nicht angefasst) | — | — |

## Widersprüche AUDIT.md ↔ Code

1. **H4 überholt:** Das Audit sagt „kein einziger Commit". Im Repo liegt `54105bf` (24.09.2026)
   mit dem gesamten Code, `.gitignore` enthält `rlviser.exe` und `settings.txt`. Offen blieb der
   Tag (lokal) und der Git-Hash in `config_used.json` (umgesetzt).
2. **N5:** `third_party/PINNED.md` verwies auf das nicht existierende `bench/cpp/CMakeLists.txt`
   und sagte „Patches am Upstream-Code: keine" — korrigiert.
3. **AUDIT_PROGRESS.md** wurde in der Aufgabenstellung als vorhanden bezeichnet, lag aber weder
   im Repo noch im Upload. Diese Datei ist neu.
4. **Neuer Upstream-Befund** (AUDIT.md 7.2): Bootstrapping an Sammelblock-Grenzen benutzte den
   Wert des nächsten Listeneintrags (anderes Spiel). Mit dem K1-Patch behoben.
5. Alle anderen Zeilen-/Dateiverweise im Audit stimmen mit dem Code und dem Upstream `ee4cc56`.

## Was nicht getestet werden konnte (ehrliche Liste, Stand Cloud-Session)

Überholt durch „Verifikation lokal" und „Was jetzt noch ungetestet ist" im Abschnitt
Review-Fixes oben; hier unverändert als Stand der Cloud-Session.

* Alle `*.ps1` (Build, Tests, Experimente, Prüfpaket): in der VM nur per PowerShell-Parser auf
  Syntax geprüft, nie ausgeführt. Erste Ausführung = Runbook Schritt 1.
* MSVC-Build mit beiden Upstream-Patches und dem neuen `rlbot_tests`-Link gegen libtorch
  (`RG_IMEXPORT` an `ComputeGAE`).
* Wirkung von K1b, K1a, H6 auf das Lernen; alle SPS-/Dauerangaben.
* `requirements.txt`-Pins gegen die tatsächliche Installation.
* H1 im echten Spiel (RLBot-Paketfolge, Phasenwechsel).

## Arbeitsprotokoll

* 25.09.2026 — Session gestartet. AUDIT.md aus dem Upload ins Repo übernommen, Herkunftstabelle
  (§0) ergänzt. Python-Tests in der VM: 51 bestanden, 3 übersprungen. Upstream geklont
  (`ee4cc56`), Linux-CPU-Build nach zwei GCC-Fixes (Patch) komplett, 39 C++-Tests grün.
* 25.09.2026 — Stufe 1 (M8, H6/H3-Schalter, M1, K2/M4/M5) und Stufe 2 (K1a, H1, M3/M7/M2, K1b)
  umgesetzt, je ein Commit. Smoke-Läufe `train_bot` auf CPU: Neustart, Kopfzeile, Metriken,
  Truncation, `extra_steps`, `save_on_exit`.
* 25.09.2026 — Schritt 2 (Experiment-Configs, Runner, Abbruchkriterien, compare), Schritt 3 (H5,
  N6/M6 dokumentiert), Schritt 4 (`run_all_checks.ps1`, Deployment-Smoke, Runbook). Stand:
  75 C++-Tests, 97 Python-Tests grün in der VM; alle PowerShell-Skripte syntaktisch geparst.
* 25.09.2026 (lokal, Trainings-PC) — Review-Befunde R1-R18 plus Nebenbefund R19 behoben, je ein
  Commit (plus vier Nachträge). Erster MSVC-cu128-Build beider Patches. K1b-Fehler (Bootstrap von
  der Reset-Obs) am echten Pfad reproduziert und korrigiert. `run_all_checks.ps1` unter
  PowerShell 5.1 komplett grün (88 C++ / 147 Python, je 2×), Smoke mit K1b-Diagnose,
  Deployment-Smoke mit Checkpoint 3907335040, Mini-Lauf `run_experiment.ps1` + `compare.py`.
  Unterwegs zwei flackernde Tests gefunden und behoben (R19, R10-Nachtrag), beide wegen der
  adressabhängigen Reihenfolge von `arena->_cars`.

## Übernahme für eine neue Session

1. `AUDIT.md` §0 (Herkunft), §6 (Roadmap) und §7 (Nachtrag, §7.2b K1b-Korrektur, §7.6
   Experiment-Design) lesen, dann die Review-Tabelle und die Statustabelle oben.
2. `git log --oneline bb7f93e..` zeigt die Commits pro Review-Befund.
3. Offene Punkte: „Was jetzt noch ungetestet ist" und „Offen / zu entscheiden" im Abschnitt
   Review-Fixes (vor allem die Duell-Remis vor Stufe 3); alles Weitere nach `LOCAL_RUNBOOK.md`.
4. Wenn lokale Ergebnisse (`results/*.zip`, `results/compare.md`) vorliegen: pro Experiment
   entscheiden (behalten / verwerfen / nachmessen) nach AUDIT.md §6 Stufe 3, dann AUDIT.md §7
   und die Roadmap mit den echten Zahlen fortschreiben; erst danach Stufe 4 (`bench_expbuffer.ps1`).
5. Linux-Entwicklung: `cmake -S . -B build/cpp_linux -G Ninja -DCMAKE_BUILD_TYPE=Release
   -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_PREFIX_PATH=<venv>/lib/python3.11/site-packages/torch/share/cmake`
   nach `git -C third_party/RLGymPPO_CPP apply` beider Patches; Meshes aus
   `<venv>/lib/python3.11/site-packages/rlgym/rocket_league/sim/collision_meshes` nach `collision_meshes/`.
