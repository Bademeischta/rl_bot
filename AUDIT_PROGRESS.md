# AUDIT_PROGRESS: Umsetzungsstand der Audit-Roadmap

Zweck: Eine neue Session (Mensch oder Agent) soll hier ohne weiteres Vorwissen einsteigen können.
Grundlage ist `AUDIT.md` (Befunde K1–N10, Roadmap Stufen 0–4). Dieses Dokument wird bei jedem
Roadmap-Punkt fortgeschrieben.

Branch: `claude/rlbot-audit-roadmap-c8t7nl` (abgezweigt von `main` @ `54105bf`).
Regel: ein Commit pro Roadmap-Punkt, ID in der Commit-Message.

## Arbeitsumgebung der Umsetzung (25.09.2026)

Cloud-Session auf einer Linux-VM (4 Kerne, 15 GB RAM, keine GPU, kein Windows, kein Rocket League,
kein `runs/`-Ordner). Alles, was Hardware braucht, ist als lokales Paket vorbereitet
(`LOCAL_RUNBOOK.md`, `tools/local/`, `tools/experiments/`).

Was in der VM läuft:

| Prüfung | Stand | Ergebnis |
|---|---|---|
| Python-Tests (`.venv`, numpy 1.26.4, torch 2.14.0+cpu, rlgym 2.0.1, rlbot 2.0.0b55, rlbot_flatbuffers 0.19.0) | vor allen Änderungen | 51 bestanden, 3 übersprungen (Policy-Parität braucht `runs/*/checkpoints`) |
| Upstream-Klon `RLGymPPO_CPP @ ee4cc56` inkl. Submodule | vorhanden unter `third_party/` (gitignored) | Patches lassen sich mit `git apply --check` prüfen |
| C++-Build unter Linux (GCC 13, CPU-libtorch aus dem pip-Wheel, Python 3.11-Header) | siehe Protokoll unten | Upstream braucht zwei GCC-Kompatibilitätsänderungen (Timer.h, gradscaler.hpp), siehe `third_party/patches/rlgympppo_cpp_gcc_compat.patch` |

**Keine Performance-Zahl aus dieser VM wird als Messwert verwendet.**

## Statustabelle

Status-Werte: `offen` · `umgesetzt (VM-getestet)` · `umgesetzt (ungetestet, lokal prüfen)` ·
`vorbereitet (lokal ausführen)` · `verworfen` · `außerhalb Scope`.

| ID | Stufe | Inhalt | Status | Commit | Test / Nachweis |
|---|---|---|---|---|---|
| H4 | 0 | Git-Commit + Tag, Git-Hash in `config_used.json` | teilweise: Commit `54105bf` existiert (vom Nutzer, nach dem Audit); Tag und Hash offen | — | Tag lokal setzen (`LOCAL_RUNBOOK.md`) |
| M8 | 1 | Metriken Episoden-Ende (`ep_end_goal`, `ep_end_timeout`, `ep_end_notouch`, `ep_end_time`, `ep_length_steps`, `scene_<name>_goal/_length`) | umgesetzt (VM-getestet) | M8 | `tests/cpp/test_metrics.cpp` (8 Tests), Linux-Build; `OnIteration` aggregiert jetzt alle `AccumAvg`-Schlüssel dynamisch |
| H6 | 1 | Seed an eigene State-Setter und Obs-Shuffle (`env.seed_envs`, Default true; false = alter zeitgeseedeter Pfad) | umgesetzt (VM-getestet) | H6 | `Seed_*`-Tests, `OBS_Shuffle_mit_Seed_*`, `EnvFactory_reicht_Seed_*`. Nicht seedbar bleiben Upstream-Teile: `RandomState`, `Arena::ResetToRandomKickoff`, SkillTracker-Seitentausch |
| M1 | 1 | `metrics.csv`: Kopfzeile aus Datei übernehmen, neue Spalten anhängen, `nan`/`inf` leer (N4), 12 signifikante Stellen | umgesetzt (VM-getestet) | M1 | `CSV_*`-Tests (7); Smoke-Lauf `train_bot` auf CPU in der VM mit Neustart (siehe Protokoll) |
| K2 | 1 | `duel.cpp` nutzt die Obs aus `gym.Reset()`/`result.obs` statt `BuildOBS` doppelt | umgesetzt (VM-getestet) | K2 | Linux-`duel` auf Smoke-Checkpoints gelaufen; `EnvFactory_Env_laeuft_100_Schritte_*` prüft, dass der Stack pro Step um genau eine Aktion wandert. Ladder-Nullmessung über die 10 echten Checkpoints: **lokal** (`LOCAL_RUNBOOK.md`) |
| M4 | 1 | Rating-Schlüssel `<lauf>/<steps>`, `ratings.json` je Lauf | umgesetzt (VM-getestet) | K2 | `test_rating_key_*`, `test_default_ratings_path_is_per_run`; `ladder.py --run` in der VM mit Linux-`duel` |
| M5 | 1 | `watch.py --run`, `test_policy_parity` nur Hauptlauf, numerisch sortiert; Build-Ordner/Lauf per `RLBOT_BUILD_DIR`/`RLBOT_PARITY_RUN` überschreibbar | umgesetzt (VM-getestet) | K2 | `test_latest_checkpoint_is_numeric_and_per_run`; Policy-Paritätstests in der VM gegen Linux-`dump_policy_actions` und Smoke-Checkpoint grün |
| K1a | 2 | `game_timeout_secs` 900 in `lucy_1v1.json` und `lucy_multimode.json`; `sanity.json` bleibt bei 120 s (Rauchtest, siehe AUDIT.md 7.5) | umgesetzt | K1a | Wirkung lokal mit `ep_end_time` (M8) pruefen: Anteil muss deutlich unter den geschaetzten 1/3 fallen |
| H1 | 2 | `PacketBuffer` in `deploy/rlbot/bot.py`: Entscheidung auf dem Paket von vor 7 Ticks; Replay/Countdown/Pause leeren Puffer und Aktions-Stack (wie ein Reset im Training) | umgesetzt (VM-getestet) | H1 | 10 neue Tests in `tests/test_bot_logic.py`: Delay 7 nach Aufwaermen, erste Entscheidung Delay 0, verpasste Ticks, grosse Luecke, Tor-Replay/Kickoff, Pause, Duplikate, Frame-Ruecksprung. Latenz im echten Spiel: **lokal** (Deployment-Smoke-Test) |
| M3 | 2 | Golden-Fixtures nicht überschreiben | offen | | |
| M7 | 2 | Obs-Größenprüfung im Bot (+ N1 toter Code) | offen | | |
| M2 | 2 | requirements pinnen + rlbot, Abgleichskript | offen | | |
| K1b | 2 | Truncation-Flag durch Upstream (Patch) | offen | | |
| H2 | 3 | `ent_coef` 0,004 — nur als Experiment-Config | offen | | |
| H3 | 3 | Slot-Shuffle: Config-Schalter `env.shuffle_slots` (Default true = altes Verhalten); Experiment-Config folgt in Schritt 2 | Schalter umgesetzt (VM-getestet), Experiment vorbereitet | H6 | `OBS_Shuffle_Slot0_Anteil_ist_ein_Drittel` bestätigt die Audit-Aussage „ein Drittel"; `OBS_ohne_Shuffle_Gegner_immer_in_Slot0` |
| K3 | 3 | Reward-Umgewichtung — nur als Experiment-Config | offen | | |
| — | 3 | `team_spirit` > 0 — Experiment-Config | offen | | |
| H5 | 4 | `exp_buffer_iterations` konfigurierbar, Benchmark-Skript | offen | | |
| M6 | 4 | Obs-Allokationen | nur als Nebeneffekt von H3 (Roadmap) | | |
| N6 | 4 | AVX-512-Zweig | offen | | |
| M9 | — | KRC `r <= 0` → `r < 0` | nicht in der Roadmap; offen | | |
| N2/N3/N5/N7/N8/N9/N10 | — | Niedrig-Punkte außerhalb der Roadmap | offen | | |

## Widersprüche AUDIT.md ↔ Code (Stand vor Umsetzung)

1. **H4 ist teilweise überholt.** Das Audit sagt „kein einziger Commit". Im Repo liegt der Commit
   `54105bf` vom 24.09.2026 mit dem gesamten Code, `.gitignore` enthält `rlviser.exe` und
   `settings.txt`. Offen bleiben nur Tag und Git-Hash in `config_used.json`.
2. **N5:** `third_party/PINNED.md` verweist auf `bench/cpp/CMakeLists.txt` (existiert nicht) und
   sagt „Patches am Upstream-Code: keine" — bestätigt; wird mit K1b aktualisiert.
3. **AUDIT_PROGRESS.md** wurde in der Aufgabenstellung als vorhanden bezeichnet, lag aber weder im
   Repo noch im Upload. Diese Datei ist neu.
4. Alle anderen Zeilen-/Dateiverweise im Audit wurden gegen den Code geprüft und stimmen
   (K1, K2, H1, H2, H3, H5, H6, M1–M9). Die Upstream-Verweise (`Gym.cpp:41/81–86/92`,
   `Match.cpp:32–38`, `ThreadAgent.cpp:139–141`, `ThreadAgentManager.cpp:55`,
   `TorchFuncs.cpp:24,36`, `PPOLearnerConfig.h:13`, `Math.cpp:59–64`) stimmen mit `ee4cc56`.

## Arbeitsprotokoll

* 25.09.2026 — Session gestartet. AUDIT.md aus dem Upload ins Repo übernommen, Herkunftstabelle
  (§0) ergänzt. Python-Tests in der VM: 51 bestanden, 3 übersprungen. Upstream geklont, Linux-Build
  gestartet.

* 25.09.2026 — M8, H6/H3-Schalter, M1 umgesetzt. Smoke-Lauf `train_bot` auf CPU in der VM (2 Threads x 2 Spiele, Netz 32x32, 2.500 + 2.500 Steps mit Neustart aus dem Checkpoint): metrics.csv hat genau eine Kopfzeile, Spalten `ep_end_*`, `ep_length_steps`, `scene_*` sind da, spaeter auftauchende Schluessel (`scene_aerial_*`) wurden hinten angehaengt. Keine Leistungszahlen aus diesem Lauf verwendet.

## Übernahme für eine neue Session

1. `AUDIT.md` §0 (Herkunft) und §6 (Roadmap) lesen, dann diese Statustabelle.
2. `git log --oneline main..` zeigt die Commits pro Roadmap-Punkt.
3. Offene Punkte stehen oben mit Status `offen`; alles mit `lokal` im Status braucht den
   Trainings-PC (siehe `LOCAL_RUNBOOK.md`).
4. Wenn lokale Ergebnisse (`results/*.zip`) vorliegen: `python tools/experiments/compare.py`
   ausführen und AUDIT.md §7 mit den echten Zahlen fortschreiben.
