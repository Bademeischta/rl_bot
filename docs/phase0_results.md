# Phase 0 — Messergebnisse und Framework-Entscheidung

Gemessen am 22./23.09.2026 auf dem Zielsystem. Alle Zahlen stammen aus
`bench/results/python_sps.csv` und `bench/results/cpp_sps.csv`, Rohdaten pro Iteration in
`bench/results/cpp_sps_raw.csv`. Die CPU-Zeilen 16×24, 12×32, 16×32 und 8×48 stammen aus dem ersten
Durchgang (`bench/results/cpp_sps_cpu_v1.csv`, noch ohne die Spalte `collect_during_learn`).

## 1. Hardware-Audit (`tools/audit.py`, Rohdaten `docs/audit.json`)

| Punkt | Wert |
|---|---|
| CPU | AMD Ryzen 7 8700F, 8C/16T, AVX2 **und AVX512** vorhanden |
| RAM | 31,7 GB |
| Frei auf C: | 130 GB |
| GPU | RTX 5070, 11,9 GB, Compute Capability **(12,0) = sm_120** |
| Torch | 2.11.0+cu128, `sm_120` im Build enthalten |
| GPU-Matmul 4096² fp32 | **22,1 TFLOPS**, kein „no kernel image" |
| OS / Python | Windows 11 (26200) / 3.11.8 |
| Repo-Ort | `C:\RLbot`, außerhalb von OneDrive |

Alle Checks PASS. **Die im Bauplan befürchtete sm_120-Blockade existiert auf PyTorch-Seite nicht**,
weil cu128-Builds sm_120 mitbringen. Der echte Reibungspunkt lag woanders, siehe Abschnitt 4.

## 2. Python-Baseline (rlgym-ppo + RLGym v2 + RocketSim)

Setup: 1v1, TickSkip 8, DefaultObs (zero_padding=3, Obs-Länge 172), LookupTable (90 Aktionen),
Netz 3×256, 100k Steps/Iteration, 1 PPO-Epoche, je 6 Mio. Steps, erste 3 Iterationen verworfen.

| n_proc | Collected SPS | Overall SPS | CPU % | RAM/Prozess | RAM gesamt |
|---|---|---|---|---|---|
| 8 | 10.169 | 9.048 | 31 % | 520 MB | 17,0 GB |
| 16 | 15.014 | 12.671 | 35 % | 521 MB | 19,8 GB |
| 24 | 20.086 | 16.253 | 36 % | 521 MB | 22,5 GB |
| **32** | **24.051** | **18.754** | 37 % | 521 MB | 25,4 GB |

Mehr als 32 Prozesse sind auf 32 GB RAM nicht sinnvoll (bei 32 Prozessen sind schon 25,4 GB belegt).
Die CPU ist dabei nur zu ~37 % ausgelastet: Der Engpass ist nicht die Simulation, sondern
Python-Inferenz und IPC im Hauptprozess.

## 3. C++ (RLGymPPO_CPP, gepinnt auf `ee4cc56`)

Gleiches Szenario, Netz 3×256, 100k Steps/Iteration, 1 Epoche. `T×G` = numThreads × numGamesPerThread.

| Build | Device | T×G | Collected SPS | Overall SPS | Collect s | Learn s | CPU % |
|---|---|---|---|---|---|---|---|
| libtorch CPU | cpu | 16×24 | 153.410 | 46.086 | 0,68 | 1,57 | 80 % |
| libtorch CPU | cpu | 16×64 | 152.987 | 47.152 | 0,68 | 1,52 | 65 % |
| libtorch CPU | cpu | 12×32 | 135.233 | 43.236 | 0,75 | 1,60 | 80 % |
| libtorch CPU | cpu | 16×32 | 115.818 | 36.555 | 0,95 | 1,99 | 88 % |
| libtorch CPU | cpu | 8×48 | 92.259 | 33.900 | 1,11 | 1,90 | 85 % |
| cu128 | **cuda** | 16×24 | 73.280 | 65.875 | 1,37 | **0,15** | 40 % |
| cu128 | cuda | 12×32 | 97.489 | 84.790 | 1,04 | 0,16 | 46 % |
| cu128 | cuda | 8×48 | 121.036 | 102.141 | 0,84 | 0,15 | 41 % |
| cu128 | cuda | 16×48 | 125.192 | 104.827 | 0,82 | 0,16 | 47 % |
| cu128 | **cuda** | **16×64** | **143.849** | **117.247** | 0,72 | 0,16 | 46 % |
| cu128 | cuda | 16×96 | 133.345 | 108.628 | 0,80 | 0,18 | 45 % |
| cu128 | cuda | 16×128 | 137.831 | 113.034 | 0,77 | 0,17 | 42 % |
| libtorch CPU + `collectionDuringLearn` | cpu | 16×64 | 70.740 | **70.190** | 0,06 | 2,08 | — |
| cu128 + `collectionDuringLearn` | cuda | 16×64 | 134.117 | 115.661 | 0,70 | 0,19 | 52 % |

Zwei Effekte:
1. **Der Learner ist auf der CPU der Flaschenhals.** Collection läuft mit ~153k SPS, aber jede
   Lernphase kostet ~1,5 s pro 100k Steps. Auf der GPU kostet sie 0,15 s, also **10× weniger**.
2. **Auf der GPU hängt die Collection an der Batch-Größe der Inferenz.** Die Agenten inferieren auf
   demselben Device wie der Learner. Mit wenigen Spielen pro Thread sind die Batches zu klein und die
   GPU-Latenz dominiert (16×24 nur 73k). Ab ~64 Spielen pro Thread ist der Effekt weg, darüber flacht es ab.
3. **`collectionDuringLearn` rettet den CPU-Fallback teilweise**: 47k → 70k, weil während der
   Lernphase weiter gesammelt wird. Auf der GPU bringt die Option nichts (115,7k gegen 117,2k),
   weil der Learner die Inferenz der Agenten dort ohnehin blockiert. Achtung: Mit dieser Option hängt
   das Programm beim Beenden reproduzierbar in „Stopping agents..." und muss abgeschossen werden
   (Upstream-Bug; im CPU-Lauf beobachtet, Messwerte waren zu dem Zeitpunkt schon geschrieben).
   **Für die GPU-Konfiguration also aus lassen.**

## 4. Toolchain-Spike: Ergebnis

| Schritt | Ergebnis |
|---|---|
| 5a CPU-libtorch (2.11.0+cpu) | **PASS**, Build mit VS Build Tools 2026 (MSVC 14.51) ohne Codeänderung |
| 5b cu128-libtorch (2.11.0+cu128) | **PASS nach einem CMake-Patch**, `Using CUDA GPU device...`, kein „no kernel image" |

Der Bauplan hat die CUDA-11.8-Angabe im README richtig als Risiko erkannt, aber die Blockade sitzt
woanders: Man kann libtorch einfach durch die cu128-Version ersetzen, der Upstream-Code braucht
dafür keine Änderung. Was tatsächlich bricht:

- `nvcc` aus CUDA 12.8 lehnt MSVC 14.51 (VS 2026) als Host-Compiler ab. Mit
  `-allow-unsupported-compiler` kommt man an die Prüfung vorbei, dann stürzt aber `cudafe++` ab
  (ACCESS_VIOLATION). nvcc ist mit VS 2026 nicht benutzbar.
- Gebraucht wird nvcc hier gar nicht: Das Projekt hat keine `.cu`-Dateien, es linkt nur die
  vorgebaute `torch_cuda.dll`. libtorch ruft in `Caffe2/public/cuda.cmake` aber unbedingt
  `enable_language(CUDA)` auf.
- Lösung: `third_party/patches/libtorch_cuda_cmake_no_enable_language.patch` macht den Aufruf über
  `-DRLBOT_SKIP_CUDA_LANGUAGE=ON` abschaltbar (anwenden: `tools\patch_libtorch_cuda.ps1`).
  **Der Patch betrifft nur libtorch, nicht RLGymPPO_CPP. Am Upstream-Code gibt es keine Änderung.**

Alternative, falls der Patch bei einem libtorch-Update bricht: VS 2022 Build Tools (Toolset v143)
installieren, das nvcc 12.8 akzeptiert.

## 5. Entscheidung (Bauplan §11 Entscheidungsbaum)

**Plan A ist bestätigt: RLGymPPO_CPP mit cu128-libtorch und GPU-Learner.**
Startkonfiguration `numThreads=16`, `numGamesPerThread=64`.

Begründung in Zahlen: 117.247 Overall SPS gegen 18.754 SPS bei der besten Python-Konfiguration,
also **6,3× schneller**. Der Fallback CPU-Learner (47k) wird nicht gebraucht, WSL2 ebenso wenig.
Die Bauplan-Schwelle „Python-SPS < 15.000 → C++ lohnt sich stark" ist mit 18,8k knapp verfehlt,
der gemessene Faktor 6,3 entscheidet die Frage aber eindeutig zugunsten von C++.

Einordnung gegen die Bauplan-Schätzungen:

| Größe | Bauplan-Schätzung | Gemessen |
|---|---|---|
| Python SPS | 15.000–25.000 | 18.754 (Overall), 24.051 (Collected) — **getroffen** |
| RLGymPPO_CPP SPS | 90.000–140.000 | 117.247 (Overall), 143.849 (Collected) — **getroffen** |
| GPU nutzbar? | „NICHT nutzbar mit README-Setup" | **nutzbar**, mit cu128-libtorch + CMake-Patch |

## 6. Wall-Clock-Modell

**Korrigiert am 23.09.2026 nach 4,1 Stunden echtem Training.** Die 117.247 SPS oben gelten für
die Benchmark-Konfiguration (Netz 3×256, DefaultOBS, einfacher Reward, 1 PPO-Epoche, keine
Skill-Eval). Die Trainings-Konfiguration ist teurer, und der Unterschied ist erheblich:

| pro Iteration (100k Steps) | Benchmark 3×256 | Training 512×3 | Training 1024/1024/512/512 |
|---|---|---|---|
| Sammeln | 0,72 s | 1,66 s | 1,89 s |
| ↳ Policy-Inferenz | — | 0,84 s | 0,85 s |
| ↳ Simulation | — | 0,47 s | 0,66 s |
| Lernen (PPO) | 0,16 s | 0,58 s | 1,85 s |
| Iteration gesamt | 0,85 s | 1,51 s | 3,74 s |
| **Effektive SPS** | **117.247** | **~68.000** | **28.762** |

**Achtung, Anlaufphase:** Die erste Messung mit 512×3 ergab 2,58 s pro Iteration (38.351 SPS)
und war zu pessimistisch, weil sie in den ersten Minuten eines frischen Laufs gemacht wurde.
Gemessen über 2,2 Mrd. Steps:

| Iterationen | Steps | Iteration | Ballkontaktrate | SPS |
|---|---|---|---|---|
| 0 – 1.761 | bis 180 Mio. | 2,20 s | 0,010 | 46.823 |
| 1.761 – 3.522 | bis 361 Mio. | 1,54 s | 0,033 | 67.019 |
| ab 14.000 | ab 1,6 Mrd. | 1,50 s | 0,028 | ~68.500 |

Der Grund ist das Training selbst: Solange die Policy den Ball kaum trifft, greift ständig der
NoTouch-Timeout, und 1024 Environments zurückzusetzen ist teuer. Mit steigender Ballkontaktrate
werden die Episoden länger und die Resets seltener. **Durchsatzmessungen also erst nach etwa
200 Mio. Steps machen** (`tools/timing_trend.py` zeigt den Verlauf).

Beide Trainingszahlen sind über Stunden gemessen, nicht hochgerechnet. Das große Netz kostet
beim Lernen exakt Faktor 2; bei der Inferenz dagegen fast nichts, weil die GPU bei diesen
kleinen Matrizen ohnehin ineffizient arbeitet (siehe Abschnitt 7). Der Hauptlauf verwendet
512×3 und liegt damit rund ein Drittel über dem großen Netz.

Ein kurzer Sanity-Lauf mit 512×3 zeigte 1,87 s pro Iteration, also deutlich mehr als die
gemessenen 2,58 s im Dauerbetrieb. Der Unterschied ist die Taktrate: Über Minuten boostet der
8700F, über Stunden bei 65 W nicht mehr. **Kurze Benchmarks überschätzen den Dauerdurchsatz
um rund ein Drittel** — das war der Hauptfehler in der ersten Prognose.

Wall-Clock = Steps ÷ effektive SPS. Bei 68.000 SPS im eingeschwungenen Zustand sind das
**5,88 Mrd. Steps pro Tag** (die Zahl enthält Lernphase und Skill-Eval bereits, ein zusätzlicher
Duty-Cycle-Faktor wäre doppelt gezählt; für Neustarts und Auswertungspausen trotzdem Luft
einplanen):

| Ziel | Netz 512×3 | Netz 1024/1024/512/512 |
|---|---|---|
| 2 Mrd. (eine Ablation) | 8,2 h | 19,3 h |
| 10 Mrd. (Seer-Größenordnung) | 1,7 Tage | 4,0 Tage |
| 30 Mrd. (Hauptlauf) | 5,1 Tage | 12,1 Tage |

Gemessen am laufenden Hauptlauf: 2,17 Mrd. Steps in 8,46 Stunden, also 71.337 SPS inklusive
der langsamen Anlaufphase.

Nachrechnen für einen laufenden Lauf: `python tools/throughput.py runs/<name>`.

## 7. Offene Punkte für Phase 1

- **Thread-Aufteilung: gemessen, nichts zu holen.** Vermutet war, dass die vielen kleinen
  Inferenz-Aufrufe (ein Aufruf je Thread und Schritt) durch größere Batches billiger werden.
  Gegenprobe mit der echten Trainings-Config (`tools/tune_threads.py`, je 2 Mio. Steps):

  | Threads × Spiele | Envs | Inferenz | Simulation | Lernen | Iteration | SPS |
  |---|---|---|---|---|---|---|
  | 16 × 64 | 1024 | 0,85 s | 0,54 s | 1,01 s | 2,78 s | 38.099 |
  | 16 × 128 | 2048 | 0,89 s | 0,57 s | 1,01 s | 2,89 s | 38.115 |
  | 8 × 128 | 1024 | 0,84 s | 0,62 s | 0,94 s | 2,76 s | 38.225 |
  | 16 × 256 | 4096 | 1,04 s | 0,56 s | 1,05 s | 3,18 s | 36.761 |

  Die Inferenzzeit bleibt konstant, obwohl sich die Zahl der Aufrufe von 781 auf 195 pro
  Iteration viertelt. Die Kosten skalieren also pro Zeile, nicht pro Aufruf: Es ist kein
  Latenzproblem, sondern schlechte GPU-Effizienz bei kleinen Matrizen. Größere Batches helfen
  nicht, sehr viele Envs schaden sogar leicht. **16 × 64 bleibt.**
- **Config-Tuning am laufenden Training** (`tools/tune_config.py`, jede Variante startet vom
  selben Checkpoint, je 2 Mio. Steps):

  | Variante | Iteration | SPS | gegen Baseline | Kostet Qualität? |
  |---|---|---|---|---|
  | Baseline (Skill-Eval alle 4 Iterationen) | 1,64 s | 62.400 | — | — |
  | **Skill-Eval alle 32 Iterationen** | 1,52 s | 67.600 | **+8,3 %** | nein, **übernommen** |
  | Skill-Tracker aus | 1,52 s | 67.900 | +8,9 % | Rating entfällt |
  | PPO-Epochen 2 → 1 | 1,37 s | 74.800 | +19,2 % | ja, halbe Updates je Sample |
  | Iterationsgröße 200k | 3,04 s | 66.400 | +5,9 % | überlappt mit der Skill-Eval |

  Der Upstream-Standard `updateInterval = 4` spielt alle vier Iterationen Eval-Spiele und kostet
  rund 8 % Durchsatz. Das Feld ist jetzt als `metrics.skill_update_interval` konfigurierbar.
- **AVX512 bringt nichts (getestet und verworfen).** Ein Build mit `/arch:AVX512` ergab in drei
  Durchläufen 69.182 / 66.205 / 70.649 SPS gegen 66.898 / 71.962 / 68.881 des Standard-Builds,
  also 68.679 gegen 69.247 im Mittel. Die Simulationszeit blieb unverändert (0,32 s).
  Die erste Einzelmessung hatte +3,4 % gezeigt — das war Rauschen.
- **Messrauschen ernst nehmen:** Zwischen identischen Läufen lagen bis zu 7 % Unterschied.
  `tools/tune_config.py` wiederholt deshalb standardmäßig dreimal und weist die Streuung aus.
  Unterschiede unter etwa 5 % sind ohne Wiederholungen nicht belastbar.
- **`minInferenceSize` ist in dieser Upstream-Version wirkungslos:** Das Feld steht in
  `LearnerConfig.h`, wird aber nirgends gelesen. Wer daran dreht, ändert nichts.
- **GPU-Auslastung täuscht:** nvidia-smi meldet 91 % bei nur 84 W (von ~250 W). Die GPU ist die
  meiste Zeit mit kleinen Kerneln beschäftigt, nicht ausgelastet. Eine schnellere Grafikkarte
  würde hier wenig bringen.
- **Netzgröße:** Alles hier mit 3×256 gemessen. Vor dem Hauptlauf mit der Zielgröße gegenmessen
  (GigaLearn wirbt damit, dass größere Netze kaum SPS kosten, für RLGymPPO_CPP ist das ungeprüft).
- **Stromaufnahme:** nicht gemessen, dafür fehlt ein Sensor-Tool. Die Bauplan-Schätzung
  (250–350 W, ~80 €/Monat bei 37 ct/kWh) bleibt unbelegt.
- **Collision-Meshes:** Es werden die von `rlgym-rocket-league` mitgelieferten Meshes benutzt
  (16 Dateien, `collision_meshes/soccar`). Ein eigener Dump aus der Rocket-League-Installation war
  nicht nötig, siehe `tools/DUMP_MESHES.md`.
