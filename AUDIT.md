# Audit: RLbot (RLGymPPO_CPP / RocketSim / RLBot v5)

Stand: 24.09.2026. Auditiert wurde der vollständige Eigencode (~5.000 Zeilen) plus die
relevanten Pfade im gepinnten Upstream `RLGymPPO_CPP @ ee4cc56`.

**Messgrundlage.** Während des Audits lief der Hauptlauf (`train_bot.exe train\configs\lucy_1v1.json`,
PID 31720). Ich habe deshalb **keine** eigenen Trainings- oder Benchmarkläufe gestartet — sie hätten
den Lauf gebremst und wären selbst durch ihn verfälscht worden. Alle Zahlen stammen aus:

* `runs/lucy_1v1/metrics.csv` — 26.789 Iterationen, 2,73 Mrd. Steps, echte Langzeitdaten
* `runs/lucy_1v1/checkpoints/2704829056/RUNNING_STATS.json`
* `bench/results/config_tuning*.csv`, `thread_tuning.csv` — deine eigenen Messreihen
* Ausgeführt habe ich nur die Testsuiten: **54 Python-Tests grün**, **39 C++-Tests grün** (je ein Lauf)

Wo eine Aussage aus einem Modell statt aus einer Messung kommt, steht es dabei.

---

## 0. Herkunft der Messwerte (Nachtrag 25.09.2026)

Ergänzt in der Umsetzungs-Session (Cloud-VM, Linux, ohne Zugriff auf den Trainings-PC, ohne
`runs/`). Kategorien:

* **lokal / Repo** — auf dem Ryzen 7 8700F + RTX 5070 gemessen, Rohdaten liegen im Repo
* **lokal / runs** — auf dem Trainings-PC aus `runs/…` gelesen, Datei ist **nicht** im Repo
  (gitignored), in der VM nicht prüfbar
* **lokal / Auditor** — vom Auditor auf dem Trainings-PC ausgeführt oder ausgelesen
* **Code** — direkt aus dem Quelltext abgeleitet, hardwareunabhängig, in der VM nachgeprüft
* **gerechnet** — aus anderen Messwerten hergeleitet (Modell oder Rechnung, keine eigene Messung)
* **VM** — in dieser Cloud-Session gemessen (nur Testergebnisse, keine Performance-Aussagen)

„Lokal nachmessen" heißt: hardwareabhängig oder aus einer Datei außerhalb des Repos, deshalb
nur auf dem Trainings-PC verifizierbar. Das lokale Paket (`tools/local/run_all_checks.ps1`,
`tools/experiments/`) sammelt genau diese Werte ein.

| Messwert (Fundstelle im Audit) | Herkunft | Rohdaten | lokal nachmessen? |
|---|---|---|---|
| 117.247 Overall SPS, Benchmark 3×256 (README, §H5) | lokal / Repo | `bench/results/cpp_sps.csv`, 23.09.2026, Kurzlauf 17 Iterationen | nein — Referenz vorhanden; gilt nur für die Benchmark-Config |
| ~68.000 SPS Training 512×3 eingeschwungen (K1, §6) | lokal / runs | `docs/phase0_results.md` §6, gerechnet aus `runs/lucy_1v1/metrics.csv` | **ja** — Basis für alle Zeitangaben; ändert sich mit K1-Sofortmaßnahme (längere Episoden) |
| 71.337 SPS effektiv, 2,17 Mrd. Steps in 8,46 h (§6) | lokal / runs | `docs/phase0_results.md` | ja, indirekt (Baseline-Experiment) |
| 62.714 / 74.763 SPS (Baseline / `ppo_epochs_1`, H5) | lokal / Repo | `bench/results/config_tuning.csv`, je **1** Lauf à 2 Mio. Steps | **ja** — Einzelmessungen, Streuung bis 7 % |
| „+10 % Durchsatz bei H5" | gerechnet | 74.763 / 67.558 (`config_tuning_skill.csv`, `skill_intervall_32`) = +10,7 %; verkettet über zwei Messreihen mit je n = 1 | **ja** — `tools/experiments/bench_expbuffer.ps1` misst 6/3/2 Updates mit Wiederholung |
| „100 Mio. Steps ≈ 25 Minuten" (Roadmap Stufe 3) | gerechnet | 100 Mio. / 68.000 SPS = 24,5 min | **ja** — hängt an der SPS-Zeile oben; `run_experiment.ps1` protokolliert die echte Dauer |
| 2,70 Mrd. Steps aktueller Checkpoint | lokal / runs | Ordnername `runs/lucy_1v1/checkpoints/2704829056` | ja — `run_all_checks.ps1` listet die vorhandenen Checkpoints |
| 2,73 Mrd. Steps / 26.789 Iterationen in `metrics.csv` | lokal / runs | `runs/lucy_1v1/metrics.csv` | ja (Nebenprodukt von `compare.py`) |
| Return-std **15,12** (K1, K3) | lokal / runs | `RUNNING_STATS.json` → `reward_running_stats.var`, `count` | **ja** — `run_experiment.ps1` liest den Wert aus dem Start-Checkpoint und schreibt ihn in die Zusammenfassung |
| **AVX-512 (N6)**: 69.182 / 66.205 / 70.649 gegen 66.898 / 71.962 / 68.881 SPS | lokal / Repo | `bench/results/config_tuning_avx{,1,2}.csv` und `config_tuning_std{,1,2}.csv`, je 2 Mio. Steps, 17 Iterationen; `docs/phase0_results.md` §7 bestätigt die Messung mit `/arch:AVX512` **auf dem Ryzen 7 8700F** | **nein** — lokal gemessen, 3 gegen 3, Differenz −0,8 % innerhalb der Streuung → Zweig wird verworfen (siehe §7) |
| „Slot 0 in einem Drittel der Fälle belegt" (H3) | Code | `env/cpp/Obs.cpp:74–81`: `std::shuffle` über 3 Gegner-Slots mit 1 echtem Gegner → Erwartung exakt 1/3; nicht gemessen | nein — in der VM per neuem C++-Test `OBS_Shuffle_Slot0_Anteil_ist_ein_Drittel` geprüft |
| Ballkontaktrate 0,0285 (+0,001 / Mrd.), Entropie 3,55 → 3,58, KL 0,0027, Clip 2,45 %, Ratio 1,0000 (Kurzfazit, H2) | lokal / runs | `metrics.csv`, letzte 2 Mrd. Steps | ja — Baseline-Experiment weist dieselben Spalten aus |
| V(s) 9,97 („Avg Val Target"), \|Advantage\| 0,40 (K1) | lokal / runs | `metrics.csv`, letzte 1 Mrd. Steps | ja (Baseline-Experiment) |
| Tor = 0,66 normalisierte Einheiten (K1) | gerechnet | 10 / 15,12 | folgt aus Return-std |
| Timeout-Anteil ≈ ⅓ der Episodenenden (K1) | gerechnet (Exponentialmodell) | keine Messung, keine Metrik vorhanden | **ja** — neue Metrik `ep_end_timeout` (M8) |
| ~5 % falsche Gradientenmasse pro Iteration (K1) | gerechnet | aus 9,97 / 0,40 / ⅓ | folgt aus Timeout-Anteil |
| Episodenlänge 2.715 Steps ≈ 181 s (K1, M8) | gerechnet | 2.049 / 0,741 aus `metrics.csv` | **ja** — neue Metrik `ep_length_steps` (M8) |
| K3-Beitragstabelle: `save_boost` 0,0934, `in_air` 0,0026, Summe 0,741 | gerechnet + lokal / runs | 0,3·√0,0969 und 0,02·0,130 aus geloggten Metriken; Summe = „Average Step Reward" (gemessen); KRC-Anteile ~0,37 / ~0,25 sind Schätzung | teilweise — Summe und die beiden Einzelterme ja, KRC-Aufteilung bleibt Schätzung |
| 5,88 Modell-Updates pro Iteration (H5) | lokal / runs, gerechnet | 157.494 `cumulative_model_updates` / 26.789 Iterationen | nein — `config_used.json` trägt künftig `exp_buffer_iterations` |
| 7 Ticks = 58 ms Beobachtungslatenz (H1) | Code | `Gym.cpp:41` `actionDelay = tickSkip − 1`; 7 / 120 s = 58,3 ms; in der VM im gepinnten Upstream nachgelesen | nein |
| 120–230 uu Versatz (H1) | gerechnet | 2.000–4.000 uu/s × 0,058 s | nein |
| Consumption 0,578 s von 1,508 s (38 %), Env Step 0,311 s (20 %) (H5, M6) | lokal / runs | `metrics.csv` | ja (Baseline-Experiment) |
| 5 Kopfzeilen und 2 `nan`-Zeilen in `metrics.csv` (M1, N4) | lokal / runs | Datei gezählt | nein — Ursache behoben (M1) |
| Skill Rating 1.196 → 1.930 (K3) | lokal / runs | `metrics.csv` | ja (Baseline-Experiment) |
| `in_air_ratio` 0,084 → 0,130, `boost_held` 0,099 → 0,0958, Episodenlänge 175 → 187 s (K3) | lokal / runs | `metrics.csv` | ja (Baseline-Experiment) |
| Thread-Tuning 38.099 / 38.115 / 38.225 / 36.761 SPS (§7 phase0) | lokal / Repo | `bench/results/thread_tuning.csv` | nein |
| Installierte Python-Versionen (numpy 1.26.4, torch 2.11.0+cu128, rlbot 2.0.0b55 …) (M2) | lokal / Auditor | `pip list` auf dem Trainings-PC | **ja** — `tools/local/check_python_versions.py` vergleicht gegen die Pins |
| 54 Python-Tests / 39 C++-Tests grün (Messgrundlage) | lokal / Auditor, je 1 Lauf | — | ja — `run_all_checks.ps1` mit `-Repeat 2` |
| Python-Tests in der Cloud-VM | VM | 51 bestanden, 3 übersprungen (Policy-Parität braucht einen Checkpoint), Stand vor den Änderungen dieser Session | — |
| C++-Tests in der Cloud-VM (Linux, CPU-libtorch 2.14 aus dem pip-Wheel) | VM | siehe `AUDIT_PROGRESS.md` | — |
| 46.432 Zeilen `tests/fixtures/obs_golden.json` (M3) | Repo | in der VM bestätigt (`wc -l`) | nein |
| Upstream-Default `entCoef = 0,005` (H2) | Code | `PPOLearnerConfig.h:13`, in der VM bestätigt | nein |
| ln 90 = 4,4998; e^3,584 ≈ 36 (H2) | gerechnet | — | nein |
| „Das Git-Repository hat keinen einzigen Commit" (H4) | lokal / Auditor, **überholt** | In der VM liegt der Stand `54105bf` (24.09.2026 11:24, „Rocket-League-RL-Bot: Training in C++ …") vor; `.gitignore` enthält `rlviser.exe` und `settings.txt`. Offen aus H4: Tag `baseline-2.7G`, Git-Hash in `config_used.json` | Tag muss lokal gesetzt werden (siehe `LOCAL_RUNBOOK.md`) |
| 2,7 Mrd. Steps ≈ 11 Stunden Rechenzeit (H4) | gerechnet | 2,7e9 / 71.337 SPS | nein |
| 47.000 SPS in der Anlaufphase, 1024 Envs (§6 phase0) | lokal / runs | `docs/phase0_results.md` | nein |

Alles, was in dieser Session an Performance-Zahlen entstanden ist, stammt aus einer geteilten
Linux-VM ohne GPU und wird **nicht** als Messwert verwendet.

---

## 1. Kurzfazit

Das Projekt ist handwerklich überdurchschnittlich: klare Schichtung (`env/` ↔ `train/` ↔ `eval/` ↔ `deploy/`),
JSON-Config mit Tippfehlerprüfung, 93 Tests mit von Hand nachgerechneten Erwartungswerten,
Golden-Fixtures für die Trainings-/Deployment-Parität, und eine Dokumentation, die Messwerte statt
Vermutungen enthält. Die Infrastruktur ist nicht das Problem.

Das Problem ist, dass **seit rund 2 Mrd. Steps nichts mehr gelernt wird**, und dass die drei Gründe
dafür alle im Reward- und Lernsignal liegen, nicht in der Geschwindigkeit. Gemessen über die letzten
2 Mrd. Steps: die Ballkontaktrate steht bei 0,0285 und steigt um +0,001 pro Mrd. Steps (also praktisch
gar nicht), die Policy-Entropie **steigt** von 3,55 auf 3,58 (bei einem Maximum von ln 90 = 4,50),
die mittlere KL pro Update liegt bei 0,0027 und die Clip-Fraction bei 2,4 %. Das ist das Bild einer
Policy, die sich nicht mehr bewegt und dabei fast so zufällig bleibt wie am Anfang.

Die drei größten Probleme:

1. **Das Torsignal ist im Reward praktisch unsichtbar.** Pro Episode sammelt ein Spieler ~2.050
   Punkte aus dichten Shaping-Termen; ein Tor ist ±10 wert. In den Einheiten, in denen der Critic
   rechnet, ist ein typischer Zustandswert 9,97 und ein Tor 0,66 — **Tore sind 6,6 % dessen wert,
   was ohnehin da ist.** Dazu ist der dichte Teil bei `team_spirit = 0` nicht zero-sum: beide
   Selbstspiel-Agenten kassieren ihn gleichzeitig, es gibt also keinen Wettbewerbsdruck aus ihm.
2. **Timeouts werden als echtes Episodenende gewertet.** NoTouch (30 s) und Spielzeit (300 s)
   landen über `Match::IsDone` in derselben `done`-Flagge wie ein Tor, und die GAE schneidet
   daraufhin das Bootstrapping ab. Der Critic bekommt am Timeout das Ziel 0 statt ~9,97 — ein
   Fehler, der **15× so groß ist wie der Wert eines Tores**.
3. **Die einzige Offline-Evaluation ist kaputt.** `eval/cpp/duel.cpp:131` baut die Beobachtung
   selbst, obwohl `gym.Step()` sie schon gebaut hat. Weil `StackedPaddedOBS::BuildOBS` den
   Aktions-Stack als Seiteneffekt fortschreibt, sieht die Policy im Duell jede Aktion doppelt.
   Damit sind `duel.exe` und die darauf aufbauende TrueSkill-Ladder (`eval/ladder.py`) wertlos —
   und genau sie sollten laut `docs/phases.md` das Gate für den Reward-Phasenwechsel liefern.

Dazu kommt ein Deployment-Risiko, das keiner der vier Paritätstests abdeckt: Das Training baut die
Beobachtung 7 Ticks (58 ms) *vor* dem Zeitpunkt, an dem die Aktion wirkt (`Gym.cpp:41,81–86`),
der RLBot-Agent dagegen aus dem aktuellen Paket. Der Bot bekommt im Spiel frischere Daten, als er
gelernt hat.

Und: **das Git-Repository hat keinen einzigen Commit.** `env/cpp/`, `eval/cpp/`, `tests/cpp/`,
`deploy/` und das Wurzel-`CMakeLists.txt` sind nicht einmal gestaged. Bei 2,7 Mrd. Steps
investierter Rechenzeit gibt es keinen Stand, auf den man zurück kann.

---

## 2. Top Quick Wins

Reihenfolge = Wirkung pro Aufwand. Keiner dieser Punkte macht bestehende Checkpoints unbrauchbar.

| # | Maßnahme | Aufwand | Erwarteter Gewinn |
|---|---|---|---|
| 1 | `git add -A && git commit` | S | Rückrollbarkeit. Vorbedingung für alles Weitere. |
| 2 | `eval/cpp/duel.cpp:131` → `result.obs` statt `BuildOBS` benutzen (K2) | S | Ladder wird überhaupt erst aussagekräftig; nebenbei halb so viel Obs-Arbeit im Duell |
| 3 | `ent_coef` 0,01 → 0,004 (H2) | S | Entropie sinkt wieder, Updates werden größer. Aktuell: KL 0,0027, Clip 2,4 % |
| 4 | Shuffle im Training abschalten (`EnvFactory.cpp:60`, `true` → `false`) (H3) | S | Im reinen 1v1 reine Kosten; entfernt zusätzlich eine Train/Deploy-Asymmetrie |
| 5 | Reward umgewichten: `goal`/`concede` hoch, dichte Terme runter (K3) | S | Das Torsignal wird überhaupt erst sichtbar |
| 6 | Metrik für den Episoden-Endgrund (Tor / NoTouch / Zeit) ergänzen (M8) | S | Ohne sie ist K1 nicht messbar, sondern nur schätzbar |
| 7 | `random_seed` auch an RocketSims Env-RNG durchreichen (H6) | S–M | Ohne das sind Vorher/Nachher-Vergleiche nicht sauber |
| 8 | Timeouts als Truncation behandeln (K1) | M | Entfernt ~5 % systematisch falsches, negatives Gradientensignal |

Punkt 6 und 7 vor Punkt 3/5 erledigen — sonst lassen sich die Wirkungen von 3 und 5 nicht belegen.

---

## 3. Findings

### KRITISCH

---

#### K1 — Timeouts werden als echtes Episodenende behandelt, das Bootstrapping fehlt

**Ort**
* `train/cpp/EnvFactory.cpp:45–52` — `NoTouchCondition` und `TimeoutCondition` stehen
  gleichberechtigt neben `GoalScoreCondition`
* `env/cpp/TimeoutCondition.h:19` — liefert schlicht `bool`
* Kette: `Match.cpp:32–38` (`IsDone` ODER-verknüpft alles zu einem `bool`) →
  `Gym.cpp:92` → `ThreadAgent.cpp:140–141` (`dones[t] = done`, `truncateds[t] = false`) →
  `ThreadAgentManager.cpp:55` (markiert nur den *letzten* Schritt eines Sammelblocks als truncated) →
  `TorchFuncs.cpp:24,36`

**Problem**

`ComputeGAE` rechnet:

```cpp
float done  = 1 - terminal[step];          // TorchFuncs.cpp:24
float trunc = 1 - truncated[step];
float pred_ret = norm_rew + gamma * next_values[step] * done;   // Zeile 36
```

Bei `terminal == 1` wird `next_values` also mit 0 multipliziert: Der Critic lernt, dass nach diesem
Zustand nichts mehr kommt. Das ist für ein Tor richtig und für einen Timeout falsch — die Welt läuft
weiter, sie wird nur nicht weiter beobachtet. Der Upstream *kann* Truncation korrekt behandeln
(das Feld `truncateds` existiert und wird in der GAE ausgewertet), aber es wird nur für den
Sammelblock-Rand gesetzt, nie für Zeitlimits. Die Unterscheidung muss aus der Env kommen, und die
liefert sie nicht.

Bemerkenswert: **Dasselbe Projekt macht es an anderer Stelle richtig.** `deploy/watch.py:109–113`
trennt sauber in `termination_cond=GoalCondition()` und
`truncation_cond=AnyCondition(NoTouchTimeoutCondition, TimeoutCondition)`. Der C++-Trainingspfad
kann es nur nicht ausdrücken.

**Auswirkung** (gerechnet aus den geloggten Werten der letzten 1 Mrd. Steps)

| Größe | Wert (normalisierte Einheiten, retStd = 15,12) | Quelle |
|---|---|---|
| typischer Zustandswert V(s) | **9,97** | `Avg Val Target`, metrics.csv |
| typischer \|Advantage\| | **0,40** | `Avg Advantage` |
| ein Tor | 10 / 15,12 = **0,66** | Config + RUNNING_STATS |
| Fehler am falsch terminierten Timeout | ≈ **9,97** | = −γ·V(s') |

Der Bootstrap-Fehler ist also **15× so groß wie der Wert eines Tores** und **25× ein typischer
Advantage**. Über die GAE klingt er mit γλ = 0,9456 pro Schritt ab und reicht damit rund 54 Schritte
(3,6 s Spielzeit) in die Vergangenheit zurück.

Grobrechnung für eine Iteration: 100.000 Steps, mittlere Episodenlänge 2.715 Steps (aus
`Average Episode Reward / Average Step Reward` = 2.049 / 0,741) → ~37 Episodenenden. Schätze ich
über ein Exponentialmodell der Episodenlänge (Mittel 181 s, harter Deckel 300 s), enden davon rund
ein Drittel im Zeit-Timeout, also ~12 pro Iteration. Injizierte Advantage-Masse:
12 · 9,97/(1−0,9456) ≈ 2.200 gegen legitime 100.000 · 0,40 = 40.000.

**→ Rund 5 % der Gradientenmasse jeder Iteration ist ein systematischer, ausschließlich negativer
Artefakt.** Er bestraft genau die langen, ballkontaktreichen Ballwechsel, die man haben will.

*Unsicherheit: Der Anteil der Timeout-Episoden (⅓) ist geschätzt, nicht gemessen — es gibt keine
Metrik für den Endgrund (siehe M8). Die Größenordnung des Fehlers pro Ereignis (9,97 gegen 0,40)
ist dagegen direkt aus den Logs abgelesen.*

**Lösung**

Saubere Variante: Die Terminal-Bedingungen müssen unterscheiden können, und die Information muss
bis in die Trajektorie durchgereicht werden. Das ist ein Eingriff in den Upstream (neues Feld in
`Gym::StepResult`, Weitergabe in `ThreadAgent`), also `third_party/patches/`-Material.

```cpp
// env/cpp/TimeoutCondition.h — vorher
class TimeoutCondition : public TerminalCondition {
    virtual bool IsTerminal(const GameState& s) { return ++steps >= maxSteps; }
};

// nachher: Marker, den die Env-Seite auswerten kann
class TimeoutCondition : public TerminalCondition {
public:
    bool firedAsTruncation = false;
    virtual void Reset(const GameState&) { steps = 0; firedAsTruncation = false; }
    virtual bool IsTerminal(const GameState&) {
        firedAsTruncation = (++steps >= maxSteps);
        return firedAsTruncation;
    }
};
```

```cpp
// RLGymSim_CPP/Gym.cpp — vorher
bool done = match->IsDone(state);
return StepResult{ obs, rewards, done, state };

// nachher
bool truncated = false;
bool done = match->IsDone(state, &truncated);   // true nur bei GoalScoreCondition
return StepResult{ obs, rewards, done, truncated, state };
```

```cpp
// ThreadAgent.cpp:140 — vorher
float done = (float)stepResult.done;
float truncated = (float)false;

// nachher
float done      = (float)(stepResult.done && !stepResult.truncated);
float truncated = (float)stepResult.truncated;
```

`ThreadAgentManager.cpp:55` muss dann von Zuweisung auf ODER umgestellt werden, damit die
Blockrand-Markierung eine bereits gesetzte Truncation nicht überschreibt.

**Billige Zwischenlösung ohne Upstream-Eingriff:** `game_timeout_secs` deutlich erhöhen (z. B. 900)
und `no_touch_timeout_secs` bei 30 lassen. Dann sind Timeouts selten genug, dass der Fehler
unter 1 % fällt. Kostet etwas Durchsatz (längere Episoden ⇒ seltenere Resets ⇒ eher schneller,
siehe `docs/phase0_results.md` §6) und ist in 10 Minuten gemacht. Als Sofortmaßnahme sinnvoll,
als Dauerlösung nicht.

**Aufwand** S (Zwischenlösung) / M (sauber) — **Gewinn** hoch: entfernt ~5 % falsches Gradientensignal,
Checkpoint-kompatibel.

---

#### K2 — `duel.exe` baut die Beobachtung doppelt und korrumpiert damit den Aktions-Stack

**Ort** `eval/cpp/duel.cpp:131` (`obsBuilder->BuildOBS(...)`) zusammen mit
`RLGymSim_CPP/Gym.cpp:91` (`match->BuildObservations(state)`) und
`env/cpp/Obs.cpp:47–50`

**Problem**

`StackedPaddedOBS::BuildOBS` ist **nicht seiteneffektfrei**:

```cpp
// env/cpp/Obs.cpp:47-50
auto& history = actionHistory[player.carId];
history.push_back(prevAction);            // <-- mutiert den Zustand des Obs-Builders
while ((int)history.size() > actionStackSize)
    history.pop_front();
```

`gym.Step()` ruft intern über `Match::BuildObservations` bereits `BuildOBS` für jeden Spieler auf.
Die Duell-Schleife ruft es ein zweites Mal auf und verwirft dabei die Obs, die `gym.Step()`
zurückgegeben hat (`duel.cpp:144`, `result.obs` wird nie gelesen). Pro Schritt und Spieler wandern
also **zwei** Einträge in die Historie.

Ablauf (nachvollziehbar durch Lesen von `Gym::Step` und der Schleife):

```
Reset          history = [0]
Schleife t=0   BuildOBS(prevActions=0)  -> [0, 0]
gym.Step       prevActions=a0, BuildOBS -> [0, 0, a0]
Schleife t=1   BuildOBS(prevActions=a0) -> [0, 0, a0, a0]
gym.Step       prevActions=a1, BuildOBS -> [0, a0, a0, a1]
Schleife t=2   BuildOBS(prevActions=a1) -> [0, a0, a0, a1, a1]
```

Die Policy sieht im Duell den Stack `[a_{t-2}, a_{t-2}, a_{t-1}, a_{t-1}, a_t]` — jede Aktion
doppelt, und der Stack reicht nur halb so weit zurück wie im Training. 40 der 257 Eingabewerte
sind systematisch falsch.

**Auswirkung**

`duel.exe` misst nicht die Spielstärke der Policy, sondern die Spielstärke der Policy unter einer
Beobachtung, die sie nie gesehen hat. Beide Seiten sind gleich betroffen, ein A-gegen-B-Vergleich
ist also nicht *zufällig*, aber er beantwortet die falsche Frage: Er rangiert Checkpoints danach,
wer mit einem kaputten Aktions-Stack am besten zurechtkommt.

Das trifft alles, was darauf aufbaut:
* `eval/ladder.py` komplett (die gesamte TrueSkill-Tabelle)
* die in `docs/phases.md:134–136` dokumentierte 15:15-Verifikation im Defense-Szenario
* das Gate, an dem laut `docs/phases.md:97–99` der Reward-Phasenwechsel hängen soll

Nebenbei: doppelte Obs-Berechnung = doppelte Kosten im Duell.

*Belegt durch Codeanalyse, nicht durch Laufzeitmessung — der Rechner war durch das Training belegt.
Der Pfad ist aber eindeutig: drei Funktionen, keine Verzweigung dazwischen.*

**Lösung**

```cpp
// eval/cpp/duel.cpp — vorher (Auszug)
gym.Reset();
GameState state = gym.prevState;
while (!done) {
    for (size_t pi = 0; pi < state.players.size(); pi++) {
        FList obs = obsBuilder->BuildOBS(player, state, match->prevActions[pi]);
        ...
    }
    auto result = gym.Step(actions);
    state = result.state;
    done  = result.done;
}
```

```cpp
// nachher: die Obs benutzen, die der Gym schon gebaut hat
FList2 obsSet = gym.Reset();
GameState state = gym.prevState;
while (!done) {
    for (size_t pi = 0; pi < state.players.size(); pi++) {
        const FList& obs = obsSet[pi];
        ...
    }
    auto result = gym.Step(actions);
    obsSet = result.obs;      // <-- war vorher ungenutzt
    state  = result.state;
    done   = result.done;
}
```

Zusätzlich empfehle ich, den Seiteneffekt sichtbar zu machen, damit das nicht wieder passiert:
`BuildOBS` als einzigen Mutator lassen und einen `const`-Pfad `PeekOBS()` anbieten, oder die
Historie explizit über eine `AdvanceHistory(carId, action)`-Methode fortschreiben, die der
`Match` aufruft.

**Aufwand** S — **Gewinn** hoch: macht die einzige Offline-Evaluation überhaupt erst brauchbar.

---

#### K3 — Reward-Balance: dichtes Shaping überlagert das Torsignal um Faktor ~200

**Ort** `train/configs/lucy_1v1.json:11–23`, `env/cpp/Rewards.cpp:29–61`

**Problem**

Die Gewichte sind einzeln plausibel und gegen Lucy-SKG begründet, aber ihr *Produkt mit der
Ereignisrate* ist es nicht. Gemessen über die letzten 2.000 Iterationen:

| Term | Gewicht | gemessener Beitrag/Step | Anteil |
|---|---|---|---|
| `save_boost` (√boost, boost_held = 0,0969) | 0,3 | 0,0934 | **13 %** |
| `offensive_potential_krc` | 1,0 | ~0,37 (gerechnet) | ~50 % |
| `dist_weighted_align_krc` | 0,5 | ~0,25 (gerechnet) | ~34 % |
| `velocity_player_to_ball` | 0,1 | ~0,02 | ~3 % |
| `in_air` (in_air_ratio = 0,130) | 0,02 | 0,0026 | 0,4 % |
| `touch_ball_to_goal_accel` | 1,0 | klein (nur bei Kontakt, 3 % der Steps) | — |
| **Summe** | | **0,741 (gemessen)** | 100 % |
| `goal` / `concede` | ±10 | — | **~0** |

Die gerechneten Zeilen sind Schätzungen, aber die Summe der Schätzungen trifft den geloggten Wert
`Average Step Reward = 0,7415` auf ~1 % genau, und die beiden direkt messbaren Zeilen
(`save_boost`, `in_air`) stammen aus geloggten Metriken. Die Aufteilung ist damit belastbar.

Pro Episode (2.715 Steps): **2.049 Punkte dichtes Shaping gegen ±10 für ein Tor.**

Zwei Verschärfungen:

1. **In 1v1 kürzt sich das Tor im geloggten Episoden-Reward exakt weg.**
   `GameInst.cpp:20` mittelt über die Spieler, und bei `goal = +10` / `concede = −10` ist die
   Summe null. Die 2.049 sind also zu 100 % Shaping — was ihr auch zu 100 % optimiert.
2. **Der dichte Teil ist nicht zero-sum.** `Rewards.cpp:59` schaltet `ZeroSumReward` nur bei
   `team_spirit > 0` ein, und `lucy_1v1.json` hat 0. Im Selbstspiel kassieren also *beide* Agenten
   gleichzeitig "nah am Ball, Richtung Tor ausgerichtet, Boost gespart". Der einzige Term, bei dem
   einer gewinnt und einer verliert, ist das Tor — und das ist der Term, der untergeht.

**Auswirkung** (gemessen über 2 Mrd. Steps)

| | 0,7 Mrd. | 2,7 Mrd. | Trend |
|---|---|---|---|
| Ballkontaktrate | 0,0282 | 0,0306 | +0,0011 / Mrd. Steps |
| Policy-Entropie | 3,548 | 3,584 | **+0,019 / Mrd. Steps (steigend)** |
| Episodenlänge | 175 s | 187 s | leicht steigend |
| `boost_held` | 0,099 | 0,0958 | flach |
| `in_air_ratio` | 0,084 | 0,130 | **+55 %** |

Der Bot wird nicht besser im Ballspielen. Er wird besser darin, in der Luft zu sein (der einzige
bedingungslose Per-Step-Reward) und ansonsten die Ballnähe zu halten. Das Skill-Rating steigt zwar
(1.196 → 1.930), aber das ist **kein Gegenbeweis**: Der Tracker (`SkillTracker.cpp:72–85`) ist
ein Elo gegen eingefrorene eigene Vorversionen mit `updateOldRatings = false`. Solange der Bot
sein 0,5 Mrd. Steps altes Ich auch nur minimal schlägt, steigt die Zahl monoton — sie kann
konstruktionsbedingt nicht plateauen. Das im Bauplan vorgesehene "TrueSkill-Plateau" als Gate
kann aus dieser Metrik nie kommen.

**Lösung**

Drei Hebel, in dieser Reihenfolge:

```jsonc
// train/configs/lucy_1v1.json — vorher
"rewards": {
  "goal": 10.0, "concede": 10.0,
  "touch_ball_to_goal_accel": 1.0,
  "offensive_potential_krc": 1.0,
  "dist_weighted_align_krc": 0.5,
  "velocity_player_to_ball": 0.1,
  "save_boost": 0.3,
  "in_air": 0.02,
  "team_spirit": 0.0
}
```

```jsonc
// nachher (Phase B aus configs/rewards_lucy.yaml, konsequenter gezogen)
"rewards": {
  "goal": 50.0, "concede": 50.0,   // 5x: ein Tor ist damit ~3,3 normalisierte Einheiten
  "touch_ball_to_goal_accel": 1.0, // bleibt: das ist der ereignisgebundene Term, den man will
  "offensive_potential_krc": 0.3,  // 1,0 -> 0,3
  "dist_weighted_align_krc": 0.1,  // 0,5 -> 0,1
  "velocity_player_to_ball": 0.03, // wie phase_b in configs/rewards_lucy.yaml
  "save_boost": 0.05,              // 0,3 -> 0,05; war mit 13% der zweitgrößte Posten
  "in_air": 0.0,                   // raus: einziger bedingungsloser Term, wird nachweislich gefarmt
  "team_spirit": 0.0
}
```

Damit läge der dichte Reward bei ~0,19/Step statt 0,741, ein Tor bei 50. Verhältnis pro Episode
~520 : 50 statt 2.049 : 10 — immer noch shaping-dominiert, aber das Torsignal ist sichtbar.

**Wichtig:** Das ist eine Verhaltensänderung und braucht laut deiner Vorgabe erst dein OK. Die
Umstellung ist Checkpoint-kompatibel (Obs-Größe unverändert), aber der Critic muss sich auf die
neue Reward-Skala neu einschwingen — `RUNNING_STATS.json` trägt einen Return-std von 15,12, der
danach nicht mehr stimmt. Rechne mit einer Delle von einigen zehn Mio. Steps.

Zweiter Hebel, unabhängig davon: **`team_spirit` auf einen kleinen Wert > 0 setzen, um überhaupt
`ZeroSumReward` zu aktivieren** — auch in 1v1, wo "Team" nur der Spieler selbst ist. Das macht den
dichten Teil kompetitiv (`ownReward − avgOpponentReward`) statt kooperativ. In 1v1 ist das exakt
"mein Shaping minus das des Gegners", was das Ballnähe-Patt auflöst. Das ist eine Ein-Zeilen-Änderung
in `Rewards.cpp:59` (`> 0` → `>= 0` mit explizitem Schalter), aber eine deutliche Verhaltensänderung.

**Aufwand** S (Gewichte) / S (zero-sum) — **Gewinn** hoch, aber Neu-Einschwingen nötig.

---

### HOCH

---

#### H1 — Deployment hat 7 Ticks weniger Beobachtungs-Latenz als das Training

**Ort** `RLGymSim_CPP/Gym.cpp:41` (`actionDelay(tickSkip - 1)`) und `Gym.cpp:81–86`
gegen `deploy/rlbot/bot.py:63–68` und `deploy/watch.py:107`

**Problem**

Das Training baut die Beobachtung absichtlich verzögert:

```cpp
// Gym.cpp:81-86, tickSkip = 8 -> actionDelay = 7
arena->Step(tickSkip - actionDelay);   // = Step(1)
state = prevState;
state.UpdateFromArena(arena);          // <-- Snapshot nach 1 Tick
arena->Step(actionDelay);              // = Step(7), gleiche Controls
```

Die Beobachtung für die nächste Entscheidung stammt also von Tick T+1, während die daraus
abgeleitete Aktion erst ab Tick T+8 wirkt: **7 Ticks = 58 ms Beobachtungslatenz**, fest eingebaut.
Das modelliert die Eingabe-/Renderlatenz eines echten Spielers.

Im Deployment gibt es diese Verzögerung nicht:

```python
# deploy/rlbot/bot.py:63-68
frame = packet.match_info.frame_num
if frame >= self.next_decision_frame:
    self.next_decision_frame = frame + TICK_SKIP
    self.current_action = self._decide(packet)   # Obs aus dem AKTUELLEN Paket
```

`deploy/watch.py:107` (`RepeatAction(action_parser, repeats=TICK_SKIP)`) hat dasselbe Problem: die
RLGym-v2-Engine liefert den Zustand nach allen 8 Ticks, ohne Zwischensnapshot.

**Auswirkung**

Der Bot bekommt im Spiel Daten, die 5–7 Ticks frischer sind als alles, was er im Training gesehen
hat. Er hat gelernt, 58 ms vorauszurechnen; jetzt rechnet er von einer bereits aktuellen Position
aus weitere 58 ms voraus und **führt seine Schüsse konsequent zu weit vor**. Bei Ballgeschwindigkeiten
um 2.000–4.000 uu/s sind 58 ms rund 120–230 uu Versatz — das ist mehr als ein Ballradius (93 uu).

Keiner der vier Paritätstests aus `docs/phases.md:157–162` fängt das ab: Sie prüfen Obs-*Layout*,
Aktionstabelle, Rotation und Inferenz — alles Zustandsabbildungen, keine Zeitbeziehungen.

*Unsicherheit: Die Trainingsseite ist eindeutig (Code oben). Für die RLBot-Seite kenne ich die
tatsächliche Pipeline-Latenz zwischen `get_output` und dem Wirksamwerden der Controls nicht; sie
liegt erfahrungsgemäß bei 1–2 Ticks, nicht bei 7. Die Richtung des Fehlers (Deployment zu frisch)
ist damit sicher, der exakte Betrag nicht. Die Richtung ist die weniger schlimme — frischere
Information ist besser als ältere — aber es bleibt Off-Distribution.*

**Lösung**

Sauberste Variante, ohne das Training anzufassen: Im Deployment den Zustand um 7 Ticks verzögern,
also die Beobachtung aus einem gepufferten Paket bauen.

```python
# deploy/rlbot/bot.py — vorher
def get_output(self, packet):
    frame = packet.match_info.frame_num
    if frame >= self.next_decision_frame:
        self.next_decision_frame = frame + TICK_SKIP
        self.current_action = self._decide(packet)
    return self._to_controller(self.current_action)
```

```python
# nachher: Entscheidung auf Basis des Pakets von vor OBS_DELAY Ticks
OBS_DELAY = TICK_SKIP - 1          # wie Gym::actionDelay

def initialize(self):
    ...
    self.packet_buffer: deque = deque(maxlen=OBS_DELAY + 1)

def get_output(self, packet):
    self.packet_buffer.append(packet)
    frame = packet.match_info.frame_num
    if frame >= self.next_decision_frame:
        self.next_decision_frame = frame + TICK_SKIP
        # ältestes Paket im Puffer = Zustand vor OBS_DELAY Ticks
        self.current_action = self._decide(self.packet_buffer[0])
    return self._to_controller(self.current_action)
```

Alternative (invasiver, aber prinzipientreuer): `actionDelay` im Training auf 0 setzen und neu
trainieren. Nicht empfohlen — die eingebaute Latenz ist realistischer, und es kostet 2,7 Mrd. Steps.

Dazu: einen Test ergänzen, der die zeitliche Beziehung prüft, nicht nur das Layout.

**Aufwand** S — **Gewinn** hoch, sobald überhaupt deployt wird. Checkpoint-kompatibel.

---

#### H2 — `ent_coef = 0,01` ist zu hoch: die Entropie steigt, die Updates stehen still

**Ort** `train/configs/lucy_1v1.json:48`, wirksam in
`PPOLearner.cpp:171` (`ppoLoss = (policyLoss - entropy * config.entCoef) * batchSizeRatio`)

**Problem**

RLGymPPO_CPP normalisiert **keine Advantages pro Batch** (nachgesehen in `PPOLearner.cpp:100–175`:
`advantages` geht roh in den Loss). Der Policy-Gradient skaliert damit direkt mit dem gemessenen
`Avg Advantage = 0,40`. Der Entropie-Bonus mit Koeffizient 0,01 arbeitet dagegen — und gewinnt.

**Auswirkung** (gemessen, letzte 2 Mrd. Steps)

| Metrik | Wert | Übliche Größenordnung |
|---|---|---|
| Policy-Entropie | 3,584 von max. 4,4998 = **79,6 %** | sinkend erwartet |
| Entropie-Trend | **+0,019 pro Mrd. Steps** | sollte fallen |
| Mean KL pro Update | **0,0027** | 0,01–0,02 |
| SB3 Clip Fraction | **2,45 %** | 10–20 % |
| Mean Ratio | 1,0000 | 1,0 |

Die effektive Perplexität ist e^3,584 = **36 von 90 Aktionen**. Nach 2,7 Mrd. Steps würfelt die
Policy also faktisch noch unter 36 Aktionen. Die KL ist um Faktor 4–7 kleiner als bei gesundem
PPO: Die Updates bewegen praktisch nichts.

Zum Vergleich: Upstream-Default ist `entCoef = 0,005` (`PPOLearnerConfig.h:13`), rlgym-ppo ebenso.
Die Config liegt beim Doppelten des üblichen Wertes.

**Lösung**

```jsonc
// train/configs/lucy_1v1.json:48 — vorher
"ent_coef": 0.01,
```
```jsonc
// nachher
"ent_coef": 0.004,
```

Das ist bewusst unter dem Upstream-Default: Weil die Advantages hier klein sind (0,40), muss der
Entropieterm entsprechend kleiner sein. Erwartetes Bild nach ~50 Mio. Steps: Entropie fällt unter
3,4, Clip-Fraction steigt über 5 %, KL Richtung 0,006.

Falls die Entropie danach zu schnell kollabiert (< 2,5), ist der Weg zurück billig — der
Checkpoint bleibt kompatibel. **Nicht gleichzeitig mit K3 ändern**, sonst ist nicht unterscheidbar,
was gewirkt hat.

**Aufwand** S — **Gewinn** hoch.

---

#### H3 — Slot-Shuffle läuft pro Step und bringt im reinen 1v1 nichts

**Ort** `env/cpp/Obs.cpp:74–81`, aktiviert in `train/cpp/EnvFactory.cpp:60` und `:73` (`true`)

**Problem**

```cpp
// env/cpp/Obs.cpp:74-81 — läuft bei JEDEM BuildOBS, also 15x pro Sekunde und Spieler
for (int i = 0; i < 2; i++) {
    FList2& list = i ? opponents : teammates;
    int target = i ? maxPlayers : maxPlayers - 1;
    while ((int)list.size() < target)
        list.push_back(FList(PLAYER_FEATURES, 0.f));
    if (shuffle)
        std::shuffle(list.begin(), list.end(), ::Math::GetRandEngine());
}
```

Bei `max_players = 3` und `mode_mix = [1,0,0]` (reines 1v1) heißt das: Der eine echte Gegner landet
bei jedem Schritt in einem zufälligen von drei 29-Werte-Blöcken, die anderen zwei sind Nullen.
Die Mitspieler-Slots sind ohnehin immer leer.

Die Absicht — Permutationsinvarianz für 2v2/3v3 — ist richtig. Aber:

* Im **1v1-Lauf gibt es nichts zu permutieren.** Es ist reine Eingabe-Randomisierung: Das Netz muss
  drei redundante Kopien des Gegner-Encoders lernen, und zu jedem Zeitpunkt trainieren zwei Drittel
  dieser Gewichte auf Nullen.
* Das **Deployment mischt nicht** (`env/obs_python.py:126–129`, bewusst so) und `duel.cpp:101`
  auch nicht. Der Gegner sitzt dort immer in Slot 0. Training und Einsatz laufen also schon heute
  auf unterschiedlichen Verteilungen — Slot 0 ist zwar im Training enthalten, aber nur in einem
  Drittel der Fälle.
* Selbst für Multi-Mode wäre **einmal pro Episode** mischen die übliche Wahl; pro Step zu mischen
  bringt keine zusätzliche Invarianz, nur zusätzliche Varianz.

**Auswirkung**

Schwer exakt zu beziffern, aber die Richtung ist klar: Die Stichprobeneffizienz für alles
Gegnerbezogene ist rund um Faktor 3 schlechter als nötig, und genau die gegnerbezogenen Fähigkeiten
(Zweikampf, Timing, Verteidigen) sind das, was in den Metriken stagniert.

*Unsicherheit: Ich kann den Effekt nicht ohne Vergleichslauf messen. Dass es im reinen 1v1 keinen
Nutzen gibt, ist dagegen keine Einschätzung, sondern folgt direkt aus der Konfiguration.*

**Lösung**

Sofort und ohne Codeänderung am Obs-Builder: Shuffle im Training ausschalten.

```cpp
// train/cpp/EnvFactory.cpp:60 — vorher
new StackedPaddedOBS(cfg.maxPlayers, cfg.actionStackSize, true),
```
```cpp
// nachher — besser noch als Config-Feld env.shuffle_slots
new StackedPaddedOBS(cfg.maxPlayers, cfg.actionStackSize, cfg.shuffleSlots),
```
mit `bool shuffleSlots = false;` in `TrainConfig` und `READ(e, seen, cfg.shuffleSlots, "shuffle_slots");`
in `Config.cpp`.

Für Phase 4 (Multi-Mode) dann nicht pro Step, sondern pro Episode mischen: eine Permutation in
`Reset()` ziehen und in `BuildOBS` anwenden.

```cpp
// env/cpp/Obs.h — Skizze
std::vector<int> slotPermTeam, slotPermOpp;
virtual void Reset(const GameState& s) {
    actionHistory.clear();
    slotPermTeam = Iota(maxPlayers - 1);  slotPermOpp = Iota(maxPlayers);
    if (shuffle) {
        std::shuffle(slotPermTeam.begin(), slotPermTeam.end(), ::Math::GetRandEngine());
        std::shuffle(slotPermOpp.begin(),  slotPermOpp.end(),  ::Math::GetRandEngine());
    }
}
```

**Checkpoint-kompatibel** — die Obs-Größe ändert sich nicht, und Slot 0 war im Training enthalten.

**Aufwand** S — **Gewinn** mittel bis hoch.

---

#### H4 — Das Git-Repository hat keinen einzigen Commit; der Kerncode ist nicht einmal gestaged

**Ort** Repo-Wurzel

**Problem**

```
$ git log --oneline
fatal: your current branch 'master' does not have any commits yet
```

Gestaged sind 46 Dateien — fast ausschließlich Phase-0-Benchmark-Artefakte. **Untracked** sind unter
anderem:

* `CMakeLists.txt` (das gesamte Build-System)
* `env/cpp/` (Rewards, Obs, State-Setter — der Kern)
* `eval/cpp/`, `tests/cpp/`
* `deploy/` komplett, `env/obs_python.py`
* `docs/phases.md`
* `rlviser.exe` (53 MB — landet bei einem unbedachten `git add -A` mit im Repo)

Zusätzlich ist `bench/cpp/CMakeLists.txt` gestaged-als-neu **und** im Arbeitsverzeichnis gelöscht
(`AD`), was `third_party/PINNED.md` widerspricht, das noch darauf verweist.

**Auswirkung**

Kein Rollback, kein `git bisect`, kein Nachvollziehen, mit welchem Code ein Checkpoint entstanden
ist. `config_used.json` neben den Checkpoints deckt nur die Config ab, nicht die Reward-Formeln
oder den Obs-Builder. Bei 2,7 Mrd. Steps (≈ 11 Stunden Rechenzeit) und mehreren geplanten
Reward-Änderungen ist das die gefährlichste einzelne Lücke im Projekt — mehr noch, weil `git status`
den Eindruck erweckt, es sei etwas versioniert.

**Lösung**

```bash
# .gitignore ergänzen
echo "rlviser.exe"  >> .gitignore
echo "*.exe"        >> .gitignore

git add -A
git commit -m "Stand 24.09.2026: Phasen 0-6, Hauptlauf bei 2,7 Mrd. Steps"
git tag baseline-2.7G
```

Danach: `git rev-parse HEAD` beim Trainingsstart mit in `config_used.json` schreiben, damit
Checkpoint und Code verknüpft sind.

```cpp
// train/cpp/main.cpp:121 — vorher
std::ofstream(runDir / "config_used.json") << cfg.ToJSONString();
```
```cpp
// nachher: Commit-Hash zum Zeitpunkt des Builds mitschreiben
// (-DRLBOT_GIT_HASH="..." in bench/cpp/build.ps1 aus `git rev-parse --short HEAD`)
json used = json::parse(cfg.ToJSONString());
used["_git"] = RLBOT_GIT_HASH;
used["_started"] = CurrentTimestamp();
std::ofstream(runDir / "config_used.json") << used.dump(2);
```

**Aufwand** S — **Gewinn** hoch (Risikovermeidung).

---

#### H5 — `ppo_epochs: 2` bedeutet tatsächlich 6 Gradientenschritte; `expBufferSize` ist versteckt

**Ort** `train/cpp/Config.cpp:223` (`lc.expBufferSize = cfg.timestepsPerIteration * 3;`)
zusammen mit `ExperienceBuffer.cpp:106–121` und `PPOLearner.cpp:103–106`

**Problem**

```cpp
// Config.cpp:223 — hartkodiert, nicht über die JSON-Config erreichbar
lc.expBufferSize = cfg.timestepsPerIteration * 3;   // = 300.000
```

```cpp
// ExperienceBuffer.cpp:115 — zerlegt den GANZEN Puffer in Batches
for (int64_t startIdx = 0; startIdx + batchSize <= curSize; startIdx += batchSize)
    result.push_back(_GetSamples(indices + startIdx, batchSize));
```

Mit `ppo_batch_size = 100.000` und Puffer 300.000 liefert das **3 Batches pro Epoche**, bei
`ppo_epochs = 2` also **6 Gradientenschritte pro Iteration** — nicht 2.

**Empirisch bestätigt:** `RUNNING_STATS.json` meldet 157.494 `cumulative_model_updates` bei
26.789 Iterationen = **5,88 Updates pro Iteration**. Genau die erwarteten 6 (minus Anlaufphase
nach jedem Neustart, in der der Puffer noch nicht voll ist).

Zwei Folgen:

1. Der Name `ppo_epochs` ist irreführend: Jedes gesammelte Sample wird 6× für einen Gradientenschritt
   benutzt (3 Iterationen im Puffer × 2 Epochen), nicht 2×. Wer den Wert tunt, verstellt
   unbemerkt das Dreifache.
2. PPO ist damit **leicht off-policy**: Zwei Drittel jedes Batches sind Daten aus den beiden
   vorherigen Iterationen. Das ist die rlgym-ppo-Idiomatik (dort ebenfalls 3× Puffer) und bei der
   gemessenen KL von 0,0027 unkritisch — aber es ist eine Abweichung von CleanRL/SB3, die strikt
   on-policy sind, und sie gehört dokumentiert.

**Auswirkung**

Der teuerste einzelne Posten im Trainingsschritt ist damit größer als gedacht: `Consumption Time`
0,578 s von 1,508 s Iterationszeit = **38 %**. Deine eigene Messung
(`bench/results/config_tuning.csv`) zeigt `ppo_epochs_1` mit 74.763 SPS gegen 62.714 SPS
Baseline; bereinigt um das unterschiedliche `skill_update_interval` (über
`config_tuning_skill.csv` verkettet) sind das **rund +10 % Durchsatz** für `ppo_epochs: 1`.

Das ist aber **kein freier Gewinn**: Es halbiert die Gradientenschritte pro Sample von 6 auf 3.
Ob das Netto hilft, hängt davon ab, ob die aktuelle Wiederverwendung nützt — bei einer KL von
0,0027 spricht einiges dafür, dass die zusätzlichen Durchläufe wenig beitragen, aber das ist
eine Hypothese, kein Messergebnis.

**Lösung**

Erst sichtbar machen, dann tunen:

```cpp
// train/cpp/Config.h — ergänzen
int64_t expBufferIterations = 3;   // Puffergröße = timestepsPerIteration * dieser Faktor
```
```cpp
// train/cpp/Config.cpp:223 — vorher
lc.expBufferSize = cfg.timestepsPerIteration * 3;
// nachher
lc.expBufferSize = cfg.timestepsPerIteration * cfg.expBufferIterations;
```

und in `ToJSONString()` mit ausgeben, damit `config_used.json` die echte Zahl trägt.

Danach als A/B mit gleichem Seed (siehe H6) messen: `ppo_epochs 2/expBufferIterations 3` (= 6 Updates,
Status quo) gegen `1/3` (= 3 Updates) gegen `2/1` (= 2 Updates, strikt on-policy).

**Aufwand** S (sichtbar machen) / M (durchmessen) — **Gewinn** mittel, bis zu +10 % Durchsatz
plus Klarheit über einen zentralen Hyperparameter.

---

#### H6 — `random_seed` erreicht die Environments nicht: Läufe sind nicht reproduzierbar

**Ort** `train/cpp/Config.cpp:220` (`lc.randomSeed = cfg.randomSeed;`) gegen
`RocketSim/src/Math/Math.cpp:59–64`

**Problem**

Der konfigurierte Seed landet an genau zwei Stellen (`Learner.cpp:60,113`): `torch::manual_seed`
und der Experience-Buffer. Der RNG, aus dem **State-Setter, `RandomState` und der Obs-Shuffle**
ziehen, ist ein ganz anderer:

```cpp
// RocketSim/src/Math/Math.cpp:59-64
std::default_random_engine& Math::GetRandEngine() {
    static thread_local auto hashThreadID = std::hash<std::thread::id>();
    static thread_local uint64_t seed = RS_CUR_MS() + hashThreadID(std::this_thread::get_id());
    static thread_local std::default_random_engine randEngine(seed);
    return randEngine;
}
```

Er ist mit der **aktuellen Uhrzeit** plus Thread-ID geseedet. (Immerhin `thread_local`, also kein
Data Race über die 16 Sammel-Threads — das ist korrekt.)

Betroffen sind `env/cpp/StateSetters.cpp:10–11,169` (alle Szenenparameter und die Szenenauswahl)
und `env/cpp/Obs.cpp:80`.

**Auswirkung**

Zwei Läufe mit identischer Config und identischem Seed erzeugen unterschiedliche Trainingsdaten.
Damit ist der von dir selbst vorgesehene Arbeitsablauf — "Lernverhalten-Fixes mit kurzem
Vergleichslauf (gleicher Seed) belegen" — nicht durchführbar: Jeder A/B-Vergleich enthält
unkontrollierte Env-Varianz. `tools/tune_config.py:70–72` weist die Streuung identischer Läufe
selbst mit ~4 % aus und warnt, dass Unterschiede darunter nichts taugen; ein Teil davon ist genau
das hier.

Nebenbei ein Widerspruch: `tests/cpp/test_config.cpp` enthält `ModusMix_ist_deterministisch` —
der Modus-Mix ist deterministisch, alles andere an der Env ist es nicht.

**Lösung**

RocketSim bietet keinen Seed-Setter für den thread-lokalen Engine. Zwei Wege:

```cpp
// Variante A (empfohlen): eigener RNG in den Projekt-State-Settern, aus dem Config-Seed abgeleitet.
// env/cpp/StateSetters.h
class WeightedStateSetter : public StateSetter {
    std::mt19937 rng;
public:
    WeightedStateSetter(const StateSetterWeights& w, uint64_t seed);
    ...
};
// EnvFactory.cpp:62 — jedes Env bekommt einen deterministischen, aber eigenen Seed
new WeightedStateSetter(cfg.states, (uint64_t)cfg.randomSeed * 1000003ull + envIndex)
```

`RandomState` aus dem Upstream zieht weiterhin aus dem globalen RNG; das lässt sich nur durch
einen kleinen Patch in `third_party/patches/` oder einen eigenen Nachbau lösen.

```cpp
// Variante B (billig, unvollständig): einen Patch, der GetRandEngine() aus einer setzbaren
// globalen Basis seedet. Ein Feld + eine Zeile in RocketSim, aber Upstream-Eingriff.
```

**Aufwand** S (Variante A für die eigenen Setter) / M (vollständig) — **Gewinn** hoch, weil es
Vorbedingung für jede belastbare Messung der Punkte K1, K3, H2, H3, H5 ist.

---

### MITTEL

---

#### M1 — `metrics.csv`: wiederholte Kopfzeilen und eine bei der ersten Iteration eingefrorene Spaltenliste

**Ort** `train/cpp/main.cpp:26–53`

**Problem**

`g_metricsColumns` ist eine globale Variable, die beim Prozessstart leer ist. Bei jedem Neustart
(also auch beim Fortsetzen eines Laufs) schreibt `AppendMetricsCSV` deshalb erneut eine Kopfzeile
**mitten in die bestehende Datei**. Gemessen: `runs/lucy_1v1/metrics.csv` enthält **5 Kopfzeilen**
bei 26.789 Datenzeilen.

Zweiter, subtilerer Teil: Die Spaltenliste wird **nur aus dem allerersten Report** gebildet
(`main.cpp:30–35`). Jeder Schlüssel, der erst später auftaucht, wird für den Rest des Laufs
stillschweigend verworfen. Konkret relevant für Phase 4: `Skill Rating 2v2` und `Skill Rating 3v3`
entstehen erst, wenn im jeweiligen Modus das erste Eval-Spiel gelaufen ist — bei
`skill_update_interval: 32` also nicht in Iteration 1.

**Auswirkung**

Drei Werkzeuge umgehen das Problem bereits einzeln
(`tools/show_metrics.py:31–32`, `tools/throughput.py:35–36`, `tools/timing_trend.py:32` über
`read_rows`) — dreimal dieselbe Zeile, statt die Ursache zu beheben. Jeder neue Konsument
(pandas, Excel, ein wandb-Import) fällt darauf herein. Der Spaltenverlust in Phase 4 würde gar
nicht auffallen.

**Lösung**

```cpp
// train/cpp/main.cpp:26-45 — vorher
bool writeHeader = g_metricsColumns.empty();
if (writeHeader)
    for (auto& pair : report.data) ...
std::ofstream fOut(g_metricsPath, std::ios::app);
```

```cpp
// nachher: Kopfzeile aus der Datei lesen, wenn es sie schon gibt; sonst neu schreiben.
// Neue Schlüssel hinten anhängen und die Kopfzeile dabei einmal neu schreiben.
static void EnsureColumns(const Report& report) {
    if (g_metricsColumns.empty() && std::filesystem::exists(g_metricsPath)) {
        std::ifstream fIn(g_metricsPath);
        std::string line;
        if (std::getline(fIn, line))
            g_metricsColumns = ParseCSVHeader(line);   // vorhandene Reihenfolge übernehmen
    }
    bool changed = false;
    for (auto& pair : report.data) {
        if (pair.first.find("_avg_total") != std::string::npos) continue;
        if (pair.first.find("_avg_count") != std::string::npos) continue;
        if (std::find(g_metricsColumns.begin(), g_metricsColumns.end(), pair.first)
                == g_metricsColumns.end()) {
            g_metricsColumns.push_back(pair.first);
            changed = true;
        }
    }
    if (changed) RewriteHeaderInPlace();   // oder: neue Datei metrics.csv.2 anlegen
}
```

Minimalvariante, falls das zu viel ist: Beim Start prüfen, ob die Datei existiert und nicht leer
ist, und dann einfach keine Kopfzeile schreiben. Das behebt 90 % des Problems in drei Zeilen.

**Aufwand** S — **Gewinn** mittel.

---

#### M2 — `requirements.txt` ohne Versions-Pins, und `rlbot` fehlt ganz

**Ort** `requirements.txt`, `README.md:23–31`

**Problem**

Keine einzige Zeile ist gepinnt:
```
torch
numpy
rlgym[rl-sim]
rlgym-ppo
rlgym-tools
wandb
psutil
nvidia-ml-py
trueskill
pyyaml
py-cpuinfo
pytest
```

Tatsächlich installiert ist: `numpy 1.26.4`, `rlgym 2.0.1`, `rlgym-rocket-league 2.0.1`,
`rlgym-api 2.0.0`, `rlgym-ppo 1.3.13`, `rlgym_tools 2.6.5`, `torch 2.11.0+cu128`,
`trueskill 0.4.5`, `wandb 0.30.0`, Python 3.11.8.

Ein Setup nach README würde heute `numpy 2.x` ziehen und damit das Env-Verhalten und möglicherweise
die Golden-Fixtures verändern.

**Gravierender:** `rlbot` und `rlbot_flatbuffers` stehen **nicht** in `requirements.txt`, werden
aber von `deploy/rlbot/bot.py:22–23` importiert. Installiert sind sie (`rlbot 2.0.0b55`,
`rlbot_flatbuffers 0.19.0`), aber ein frisches Setup nach README erzeugt eine Umgebung, in der
Phase 6 nicht startet. Die Tests fallen das nicht auf, weil `tests/test_bot_logic.py:7–8` beide
Pakete per `pytest.importorskip` überspringt — auf einer frischen Maschine würden die 8 Bot-Tests
stillschweigend übersprungen und die Suite bliebe grün.

**Lösung**

```
# requirements.txt — nachher
torch==2.11.0+cu128        # separat aus dem cu128-Index, siehe README
numpy==1.26.4
rlgym==2.0.1
rlgym-rocket-league==2.0.1
rlgym-api==2.0.0
rlgym-ppo==1.3.13
rlgym-tools==2.6.5
rlbot==2.0.0b55            # fehlte; deploy/rlbot/bot.py braucht es
rlbot_flatbuffers==0.19.0  # fehlte
wandb==0.30.0
psutil==7.2.2
nvidia-ml-py==13.610.43
trueskill==0.4.5
pyyaml==6.0.3
py-cpuinfo==9.0.0
pytest==9.1.1
```

Erzeugen mit `.\.venv\Scripts\python -m pip freeze > requirements.lock.txt` und die
Direktabhängigkeiten davon ableiten.

Zusätzlich: In `tools/run_all_tests.ps1` einen Lauf mit `-p no:randomly --strict-markers` und
einer Prüfung ergänzen, dass die erwartete Testanzahl (54) erreicht wird — sonst maskiert
`importorskip` fehlende Abhängigkeiten.

**Aufwand** S — **Gewinn** mittel.

---

#### M3 — Die Golden-Fixtures werden vor jedem Testlauf neu erzeugt

**Ort** `tools/run_all_tests.ps1:21–23`

**Problem**

```powershell
Write-Host "`n--- Golden-Fixtures neu erzeugen ---"
& "$Build\dump_obs.exe" "$Root\tests\fixtures\obs_golden.json" 30 | Select-Object -Last 1
```

Die 46.432 Zeilen große Referenzdatei wird bei **jedem** Testlauf aus dem aktuellen C++-Binary
überschrieben. Danach prüft `tests/test_obs_parity.py`, ob Python zu dieser frisch erzeugten Datei
passt.

Als **Paritätstest** (C++ ↔ Python) ist das vertretbar. Als **Regressionstest** ist es wirkungslos:
Ändert jemand `env/cpp/Obs.cpp` — sei es die Reihenfolge, ein Normierungsfaktor oder ein Feature —
wandert die Referenz einfach mit. Der Test bleibt grün, und es gibt **keinen einzigen Test, der
merkt, dass das Obs-Layout sich geändert hat**.

**Auswirkung**

Genau diese Änderung ist die, die alle vorhandenen Checkpoints unbrauchbar macht (das Netz erwartet
257 Eingaben in fester Bedeutung). Bei 2,7 Mrd. Steps investierter Rechenzeit ist das die teuerste
mögliche unbemerkte Regression. `docs/phases.md:159` führt den Test als Absicherung gegen
"Obs-Layout weicht ab" — gegen Abweichung *zwischen* C++ und Python schützt er, gegen Abweichung
*über die Zeit* nicht.

**Lösung**

Zwei Dateien statt einer:

```powershell
# tools/run_all_tests.ps1 — nachher
# Fixtures NICHT überschreiben. Neu erzeugen nur in eine temporäre Datei und vergleichen.
& "$Build\dump_obs.exe" "$env:TEMP\obs_check.json" 30 | Select-Object -Last 1
$ref = Get-FileHash "$Root\tests\fixtures\obs_golden.json" -Algorithm SHA256
$new = Get-FileHash "$env:TEMP\obs_check.json" -Algorithm SHA256
if ($ref.Hash -ne $new.Hash) {
    Write-Host "Obs-Layout hat sich geaendert - bestehende Checkpoints sind INKOMPATIBEL." -ForegroundColor Red
    Write-Host "Wenn das beabsichtigt ist: tools\update_golden.ps1 ausfuehren." -ForegroundColor Yellow
    $failed++
}
```

Plus ein eigenes `tools/update_golden.ps1`, das die Referenz bewusst aktualisiert. Dann ist das
Überschreiben eine Entscheidung und kein Nebeneffekt — und `git diff` zeigt sie.

Zusätzlich sinnvoll: eine Prüfsumme des Obs-Layouts (`GetOBSSize()` + die Feature-Reihenfolge)
in jeden Checkpoint schreiben, damit `load_policy` einen inkompatiblen Checkpoint beim Laden
ablehnt statt still falsch zu spielen.

**Aufwand** S — **Gewinn** mittel (Risikovermeidung, aber teures Risiko).

---

#### M4 — `eval/ratings.json` ist global; Checkpoint-Namen kollidieren zwischen Läufen

**Ort** `eval/ladder.py:24,79–84,133–145`

**Problem**

```python
RATINGS_PATH = ROOT / "eval" / "ratings.json"
...
name_a=path_a.parent.name if path_a.name.endswith(".lt") else path_a.name
```

Der Rating-Schlüssel ist der Ordnername eines Checkpoints, also die reine Step-Zahl
(`"2704829056"`). Die Datei ist lauf-übergreifend. Zwei Läufe können denselben Step-Stand haben —
`runs/sanity/checkpoints/5053568` und ein gleich benannter Ordner aus einem anderen Lauf teilen
sich dann denselben Eintrag. Vorhanden ist bereits sowohl `runs/sanity/ratings.json` als auch der
Default `eval/ratings.json`, also zwei konkurrierende Ablagen.

Besonders relevant, weil `runs/archive/lucy_1v1_net1024/` Checkpoints mit **anderer Netzgröße**
enthält, die laut `docs/phases.md:110–112` nicht vergleichbar sind.

**Lösung**

```python
# eval/ladder.py — vorher
name_a=path_a.parent.name if path_a.name.endswith(".lt") else path_a.name,

# nachher: Lauf-Name mit in den Schlüssel
def _rating_key(path: Path) -> str:
    ckpt = path.parent if path.name.endswith(".lt") else path
    run = ckpt.parent.parent.name        # runs/<lauf>/checkpoints/<steps>
    return f"{run}/{ckpt.name}"
```

und `--ratings` per Default auf `<run>/ratings.json` legen statt auf `eval/ratings.json`.

**Aufwand** S — **Gewinn** mittel. (Sinnvoll erst **nach** K2 — vorher ist die Ladder ohnehin
nicht aussagekräftig.)

---

#### M5 — Checkpoint-Auswahl greift über alle Läufe hinweg, einmal sogar lexikografisch

**Ort** `deploy/watch.py:71–76`, `tests/test_policy_parity.py:23–25`

**Problem**

```python
# deploy/watch.py:72-76 — numerisch sortiert, aber über ALLE Läufe
candidates = sorted(ROOT.glob("runs/*/checkpoints/*/PPO_POLICY.lt"),
                    key=lambda p: int(p.parent.name))
```
```python
# tests/test_policy_parity.py:24 — lexikografisch: "999" > "1000"
candidates = sorted(ROOT.glob("runs/*/checkpoints/*/PPO_POLICY.lt"))
```

Beide greifen quer über `runs/lucy_1v1`, `runs/sanity`, `runs/tune_*` und `runs/archive`. `watch.py`
kann damit einen Checkpoint aus `runs/archive/lucy_1v1_net1024/` erwischen (Netz 1024/1024/512/512
statt 512×3) — das lädt zwar, ist aber ein anderer Bot als gedacht. Der Test ist zusätzlich
lexikografisch sortiert und wählt deshalb einen willkürlichen Checkpoint; da er nur Obs-Größe (257)
und Aktionszahl (90) prüft, fällt das nie auf.

**Lösung**

```python
# deploy/watch.py — nachher
ap.add_argument("--run", type=Path, default=ROOT / "runs" / "lucy_1v1",
                help="Lauf, aus dem der neueste Checkpoint genommen wird")

def latest_checkpoint(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "checkpoints").glob("*/PPO_POLICY.lt"),
                        key=lambda p: int(p.parent.name))
    if not candidates:
        raise SystemExit(f"Kein Checkpoint in {run_dir}")
    return candidates[-1]
```
```python
# tests/test_policy_parity.py — nachher
def _find_checkpoint() -> Path | None:
    c = sorted((ROOT / "runs" / "lucy_1v1" / "checkpoints").glob("*/PPO_POLICY.lt"),
               key=lambda p: int(p.parent.name))
    return c[-1] if c else None
```

**Aufwand** S — **Gewinn** mittel.

---

#### M6 — Der Obs-Builder alloziert bei jedem Schritt neu

**Ort** `env/cpp/Obs.cpp:60–86`

**Problem**

Pro Spieler und Schritt entstehen:
* `FList2 teammates, opponents` — zwei `std::vector<std::vector<float>>`
* ein `FList playerObs` je anderem Spieler (`Obs.cpp:64`)
* `FList(PLAYER_FEATURES, 0.f)` für **jeden leeren Slot** (`Obs.cpp:78`) — im 1v1 bei
  `max_players = 3` sind das 4 Heap-Allokationen à 29 floats, jeden Schritt, jeden Spieler

Bei 68.000 Player-Steps/s sind das grob 270.000 Allokationen pro Sekunde allein für Nullblöcke,
die immer identisch sind.

**Auswirkung**

Nicht gemessen — ich konnte nicht profilen, ohne den laufenden Lauf zu stören. Zur Einordnung:
`Env Step Time` (Simulation **plus** Obs-Bau) liegt bei 0,311 s von 1,508 s Iterationszeit, also
20 %. Der Obs-Anteil daran ist unbekannt; realistisch sind einstellige Prozent Gesamtgewinn, nicht
mehr. Es ist ein Aufräum-, kein Performance-Notfall.

**Lösung**

```cpp
// env/cpp/Obs.cpp — vorher
while ((int)list.size() < target)
    list.push_back(FList(PLAYER_FEATURES, 0.f));
```
```cpp
// nachher: eine statische Nullzeile, plus reservierte Puffer als Member
static const FList ZERO_BLOCK(PLAYER_FEATURES, 0.f);
while ((int)list.size() < target)
    list.push_back(ZERO_BLOCK);          // kopiert, aber ohne Initialisierungsschleife
```

Deutlich wirksamer wäre, ganz ohne Zwischen-`FList` direkt in `result` zu schreiben und die
Slot-Permutation nur als Indexliste zu führen (passt gut zu H3, wo die Permutation ohnehin pro
Episode fixiert wird):

```cpp
// Skizze: result vorab auf GetOBSSize() bringen, dann per Offset befüllen
result.resize(GetOBSSize());
size_t off = BALL_FEATURES + CommonValues::BOOST_LOCATIONS_AMOUNT;
...
for (int slot = 0; slot < maxPlayers; slot++) {
    size_t base = oppBase + slotPermOpp[slot] * PLAYER_FEATURES;
    if (slot < (int)opponents.size())
        WritePlayerAt(result, base, opponents[slot], ball, inv);
    // sonst: bleibt 0, weil result vorinitialisiert ist -> keine Allokation
}
```

**Aufwand** S (Nullblock) / M (Umbau) — **Gewinn** niedrig bis mittel.

---

#### M7 — Der RLBot-Agent prüft die Obs-Größe nicht gegen die geladene Policy

**Ort** `deploy/rlbot/bot.py:37–57`

**Problem**

```python
self.policy = load_policy(policy_path)
expected = build_obs.__module__       # nur zur Klarheit in Fehlermeldungen
self.logger.info(f"Policy geladen: {self.policy.meta.layer_sizes}, "
                 f"Obs {self.policy.meta.obs_size}, Aktionen {self.policy.meta.action_count}")
...
del expected
```

`meta.obs_size` wird geloggt, aber nie gegen `obs_size(MAX_PLAYERS, ACTION_STACK)` geprüft. Die
Konstanten `MAX_PLAYERS = 3` und `ACTION_STACK = 5` stehen hartkodiert in `bot.py:31–32`, unabhängig
von der Config, mit der trainiert wurde. Exportiert jemand einen Checkpoint mit
`action_stack_size: 3`, fällt das erst beim ersten `matmul` im laufenden Spiel auf.

`eval/cpp/duel.cpp:90–97` macht diese Prüfung übrigens korrekt — nur der Produktivpfad nicht.

**Lösung**

```python
# deploy/rlbot/bot.py:44 — nachher
self.policy = load_policy(policy_path)
expected_obs = obs_size(MAX_PLAYERS, ACTION_STACK)
if self.policy.meta.obs_size != expected_obs:
    raise ValueError(
        f"Policy erwartet Obs-Größe {self.policy.meta.obs_size}, der Obs-Builder liefert "
        f"{expected_obs}. Passen MAX_PLAYERS={MAX_PLAYERS}/ACTION_STACK={ACTION_STACK} "
        f"zur Trainings-Config?")
if self.policy.meta.action_count != len(LOOKUP_TABLE):
    raise ValueError(f"Policy hat {self.policy.meta.action_count} Aktionen, "
                     f"die Tabelle {len(LOOKUP_TABLE)}")
```

Und die beiden toten Zeilen (`expected = ...` / `del expected`) entfernen.

Besser noch: `max_players` und `action_stack_size` beim Export in `policy.json` mitschreiben
(`tools/export_policy.py` schreibt die Datei ohnehin) und `bot.py` sie von dort lesen lassen,
statt sie zu duplizieren.

**Aufwand** S — **Gewinn** mittel.

---

#### M8 — Es fehlt jede Metrik zum Episoden-Ende

**Ort** `train/cpp/main.cpp:56–67` (`OnStep`)

**Problem**

Geloggt werden `player_speed`, `ball_touch_ratio`, `in_air_ratio`, `boost_held`,
`supersonic_ratio`, `ball_speed`, `ball_height`. **Nicht** geloggt: wie eine Episode endet
(Tor / NoTouch-Timeout / Zeit-Timeout), wie lang sie war, wie viele Tore fallen, und welcher
State-Setter sie gestartet hat — obwohl `WeightedStateSetter::lastPicked` (`StateSetters.h:68`)
genau dafür vorgesehen ist und nirgends gelesen wird.

**Auswirkung**

* K1 lässt sich nicht messen, nur schätzen (siehe dort).
* Die Wirkung von K3 (Reward-Umgewichtung) wäre an "Tore pro Episode" sofort ablesbar — diese
  Zahl existiert nicht.
* Ob die Mechanik-Szenen (`aerial`, `dribble`, `wall_play`, `recovery`, `defense`) etwas bringen,
  ist nicht auswertbar, weil kein Reward und keine Episodenlänge je Szene vorliegt.
* Die Episodenlänge lässt sich nur indirekt aus `Average Episode Reward / Average Step Reward`
  zurückrechnen (so habe ich die 181 s bestimmt) — das bricht, sobald sich die Reward-Gewichte
  ändern.

**Lösung**

```cpp
// train/cpp/main.cpp:56 — nachher
static void OnStep(GameInst* gameInst, const Gym::StepResult& stepResult, Report& gameMetrics) {
    auto& state = stepResult.state;
    for (auto& player : state.players) { /* wie bisher */ }
    gameMetrics.AccumAvg("ball_speed", state.ball.vel.Length());
    gameMetrics.AccumAvg("ball_height", state.ball.pos.z);

    if (stepResult.done) {
        bool scored = RLGSC::Math::IsBallScored(state.ball.pos);
        gameMetrics.AccumAvg("ep_end_goal",       scored);
        gameMetrics.AccumAvg("ep_end_timeout",   !scored);
        gameMetrics.AccumAvg("ep_length_steps",   gameInst->totalSteps - lastEnd);  // pro GameInst mitführen
        auto* setter = dynamic_cast<RLbot::WeightedStateSetter*>(gameInst->match->stateSetter);
        if (setter)
            gameMetrics.AccumAvg("scene_" + setter->names[setter->lastPicked] + "_goal", scored);
    }
}
```

`ep_end_goal` ist die eine Zahl, die man braucht, um K1 und K3 zu beurteilen. Sie kostet nichts.

**Aufwand** S — **Gewinn** mittel, aber Vorbedingung für die Bewertung der kritischen Punkte.

---

#### M9 — `KRCReward` behandelt eine Komponente von exakt 0 als negativ

**Ort** `env/cpp/Rewards.h:39–49`

**Problem**

```cpp
for (auto f : funcs) {
    float r = f->GetReward(player, state, prevAction);
    if (r <= 0) allPositive = false;      // <-- r == 0 zählt als "nicht positiv"
    prod *= std::abs((double)r);
}
float mag = (float)std::pow(prod, 1.0 / (double)funcs.size());
return allPositive ? mag : -mag;
```

Bei `r == 0` wird `prod = 0`, also `mag = 0`, und das Ergebnis ist `-0.0f`. Numerisch ist das
folgenlos (`-0.0f == 0.0f`), aber die Bedingung ist trotzdem falsch formuliert: Gemeint ist
"irgendeine Komponente ist negativ", geschrieben steht "irgendeine Komponente ist nicht positiv".

Praktisch relevant wird das, sobald jemand die KRC um eine Komponente erweitert, die legitim
0 werden kann, ohne dass das Produkt 0 wird — etwa über eine additive Konstante oder einen
Clamp auf ein Minimum. Der Test `KRC_eine_Null_Komponente_macht_alles_Null`
(`tests/cpp/test_rewards.cpp:85–90`) deckt nur den heutigen Fall ab.

**Lösung**

```cpp
// nachher
if (r < 0) anyNegative = true;
...
return anyNegative ? -mag : mag;
```

**Aufwand** S — **Gewinn** niedrig (Robustheit). Ich führe es hier statt unter "Niedrig",
weil KRC der zentrale Reward-Baustein ist und jede Änderung daran das Lernverhalten trifft.

---

### NIEDRIG

| # | Ort | Problem | Lösung | Aufwand |
|---|---|---|---|---|
| N1 | `deploy/rlbot/bot.py:45,57` | Toter Code: `expected = build_obs.__module__` … `del expected` | Beide Zeilen löschen (siehe M7) | S |
| N2 | `deploy/policy.py:103` | `torch.load(..., weights_only=False)` führt beim Laden beliebigen Code aus. Die Datei ist selbst erzeugt, aber der Parameter ist unnötig: Der Blob ist ein Dict aus Tensoren plus ein Meta-Dict. | `weights_only=True` und `meta` als reines Dict laden | S |
| N3 | `deploy/policy.py:68`, `eval/cpp/policy_io.h:25` | `torch.jit.load` ist in torch 2.11 deprecated (die Warnung erscheint in 4 Tests). Der `.lt`-Lesepfad bricht in einer künftigen Version. | Mittelfristig auf `torch.export` umstellen oder die Gewichte beim Checkpointing zusätzlich als reines `state_dict` ablegen | M |
| N4 | `runs/lucy_1v1/metrics.csv` | 2 von 26.789 Zeilen enthalten `nan` in `Average Episode Reward` (Iteration ohne abgeschlossene Episode) | In `AppendMetricsCSV` nicht-endliche Werte als leeres Feld schreiben | S |
| N5 | `third_party/PINNED.md:20` | Verweist auf `bench/cpp/CMakeLists.txt`, die gelöscht ist (`git status`: `AD`); der Build läuft jetzt über das Wurzel-`CMakeLists.txt`. Außerdem steht dort "Patches am Upstream-Code: keine", obwohl `patches/libtorch_cuda_cmake_no_enable_language.patch` existiert (der betrifft libtorch, nicht RLGymPPO_CPP — das gehört präzisiert) | Zeile aktualisieren | S |
| N6 | `build/cpp_cu128_avx512/`, `build_avx512.log`, `build_t.log` | Der AVX512-Build hat **keinen messbaren Nutzen**: 3 Läufe AVX512 (69.182 / 66.205 / 70.649, Mittel 68.679 SPS) gegen 3 Läufe Standard (66.898 / 71.962 / 68.881, Mittel 69.247 SPS) — Standard ist minimal *schneller*, der Unterschied liegt klar in der von `tune_config.py` selbst ausgewiesenen ~4-%-Streuung. Das Build-Verzeichnis ist zudem leer (kein `.exe`), die Logs liegen im Repo-Root. | AVX512-Zweig verwerfen, Verzeichnis und Logs löschen, Ergebnis in `docs/phase0_results.md` festhalten | S |
| N7 | `deploy/rlbot/bot.toml:4–5` | `run_command` nutzt einen relativen Pfad (`..\..\.venv\Scripts\python.exe`), der davon abhängt, dass RLBot mit `bot.toml`s Ordner als cwd startet | Funktioniert heute; einen Kommentar ergänzen oder auf einen absoluten Pfad umstellen | S |
| N8 | `deploy/watch.py:92` | `action_parser._lookup_table = LOOKUP_TABLE.copy()` überschreibt ein privates Attribut von `LookupTableAction`. Bricht still, wenn rlgym die Tabelle künftig anders ableitet. | Eigene `ActionParser`-Unterklasse statt Monkeypatch; mindestens eine Assertion auf die vorherige Form | S |
| N9 | Repo-Wurzel | `settings.txt` (RLViser-Einstellungen, u. a. `game_speed=2.9`) liegt unversioniert und unkommentiert im Wurzelverzeichnis; nichts im Code referenziert die Datei | Nach `deploy/` verschieben oder in `.gitignore` aufnehmen und in der README erwähnen | S |
| N10 | `env/factory.py`, `tests/test_smoke.py` | Der Smoke-Test prüft die **Python**-Env aus Phase 0 (DefaultObs, 172 Werte), nicht die tatsächliche Trainings-Env (`EnvFactory` → `StackedPaddedOBS`, 257 Werte). Es gibt keinen einzigen Test, der ein echtes Trainings-Env erzeugt und Schritte darin macht — genau deshalb konnte K2 unbemerkt bleiben. | Einen C++-Test ergänzen, der `EnvFactory::Create()` aufruft, 100 Schritte macht und prüft, dass der Aktions-Stack pro Schritt genau um eine Aktion weiterwandert | M |

---

## 4. Abweichungen gegenüber CleanRL / Stable-Baselines3

Wie im Auftrag gefordert, hier die Unterschiede zur Referenz-PPO-Implementierung. Nicht jede
Abweichung ist ein Fehler — die meisten sind bewusste rlgym-ppo-Idiomatik.

| Punkt | CleanRL / SB3 | RLGymPPO_CPP | Bewertung |
|---|---|---|---|
| Advantage-Normalisierung pro Minibatch | ja (SB3 `normalize_advantage=True`) | **nein** (`PPOLearner.cpp:100–175`) | Ersetzt durch Reward-Skalierung mit dem laufenden Return-std. Funktioniert, macht aber `ent_coef` skalenabhängig → Ursache von **H2** |
| Truncation ≠ Termination | ja (Gymnasium `TimeLimit.truncated`) | **nein** auf Env-Ebene | **K1** |
| Rollout strikt on-policy | ja (Puffer = Rollout) | nein: Puffer = 3 × Rollout | **H5**, rlgym-ppo-Idiomatik |
| Gradient-Clipping | ja, 0,5 | ja, 0,5 (`PPOLearner.cpp:274,276`) | in Ordnung |
| LR-Schedule | optional linear | **keiner** | konstant 2e-4; für einen Mehrtageslauf wäre ein Abfall sinnvoll |
| Value-Loss-Clipping | optional (Default aus) | nein | in Ordnung |
| Gemeinsamer Optimizer für Policy und Critic | ja (SB3) | getrennte Adam-Instanzen | in Ordnung, sogar flexibler (getrennte LRs) |
| Optimizer-State im Checkpoint | ja | ja (`PPOLearner.cpp:436–466`) | in Ordnung, Fortsetzen funktioniert nachweislich |
| Obs-Normalisierung | SB3 `VecNormalize` | **nicht implementiert** (`LearnerConfig.h:35`, `Learner.cpp:34–35` bricht bei `true` ab) | Hier unkritisch, weil `StackedPaddedOBS` bereits fest normiert (`Obs.h:49–52`) |
| `minInferenceSize` | — | Feld existiert, wird **nirgends gelesen** (verifiziert per grep) | `docs/phase0_results.md:189–190` sagt das bereits richtig |

---

## 5. Was gut ist

Der Vollständigkeit halber, weil ein Audit sonst ein schiefes Bild gibt:

* **Die Trainings-/Deployment-Parität ist ernsthaft abgesichert.** Vier Golden-Tests über
  120 Obs-Vektoren, 90 Aktionen, 47 Rotationsfälle und 120 Policy-Auswertungen, alle mit Abweichung
  0 bzw. < 1e-5. Die drei RLBot-API-Fallstricke (Boost-Timer-Richtung, Pad-Reihenfolge,
  `dodge_timeout == -1` in zwei Bedeutungen) sind erkannt, gekapselt und getestet. Ich habe die
  Boost-Pad-Inversion (`GameState.cpp:86` gegen `obs_python.py:144–147`) und die
  `hasFlip`-Herleitung (`PlayerData.cpp:28–30` gegen `packet_adapter.py:69–83`, inklusive des
  Falls "ohne Sprung in der Luft") gegen den Upstream nachgeprüft — beide stimmen.
* **Die Config-Behandlung ist vorbildlich:** unbekannte Felder sind ein Fehler, Plausibilitäts-
  prüfungen mit verständlichen Meldungen, Roundtrip-Test, und die benutzte Config landet neben
  den Checkpoints.
* **Die Tests rechnen nach, statt Ausgaben festzuschreiben.** `test_rewards.cpp` prüft die
  KRC gegen von Hand gerechnete geometrische Mittel und die Distanzformel gegen `exp(-0.5)` —
  das ist die richtige Art, eine Reward-Funktion zu testen.
* **Die Dokumentation korrigiert sich selbst.** `docs/phase0_results.md:118,131–154` revidiert
  die eigene Durchsatzprognose nach echten Messungen und benennt den Fehler ("Kurze Benchmarks
  überschätzen den Dauerdurchsatz um rund ein Drittel"). Das ist selten.
* **Das Größte-Reste-Verfahren für den Modus-Mix** (`EnvFactory.cpp:24–38`) ist die richtige Wahl
  und getestet, inklusive kleiner Anteile bei wenigen Envs.
* **Die Thread-Aufteilung wurde gemessen statt geraten**, mit dem ehrlichen Ergebnis "nichts zu
  holen" (`docs/phase0_results.md:174–188`).

---

## 6. Roadmap

Reihenfolge nach Abhängigkeiten. **Zuerst Messbarkeit, dann Korrektheit, dann Lernsignal, dann
Geschwindigkeit** — in dieser Reihenfolge, weil sonst nicht feststellbar ist, ob eine Änderung
etwas gebracht hat.

### Stufe 0 — Absichern (vor allem anderen)
1. **H4** Git-Commit + Tag. 10 Minuten.
2. Aktuellen Checkpoint (2,70 Mrd. Steps) außerhalb von `runs/` sichern, damit die geplanten
   Änderungen einen definierten Rückfallpunkt haben.

### Stufe 1 — Messbar machen
3. **M8** Metriken `ep_end_goal`, `ep_end_timeout`, `ep_length_steps`, Szenen-Aufschlüsselung.
4. **H6** Seed an die eigenen State-Setter durchreichen.
5. **M1** Kopfzeilen-Problem an der Quelle beheben.
6. **K2** `duel.cpp` reparieren, **M4** Rating-Schlüssel, **M5** Checkpoint-Auswahl.
   → Danach einmal die Ladder über die vorhandenen 10 Checkpoints laufen lassen. Das ist die
   **Nullmessung**, gegen die alles Folgende verglichen wird.

Nach Stufe 1 weißt du zum ersten Mal, wie viele Episoden in einem Tor enden und wie die
Checkpoints tatsächlich zueinander stehen. Ohne das sind alle folgenden Schritte Blindflug.

### Stufe 2 — Korrektheit (Checkpoint-kompatibel)
7. **K1** Sofortmaßnahme: `game_timeout_secs` auf 900 erhöhen. Mit der neuen Metrik aus Schritt 3
   direkt überprüfbar: Der Timeout-Anteil muss deutlich fallen.
8. **H1** Paketpuffer im RLBot-Agenten (7 Ticks). Nur relevant, wenn deployt wird — aber billig
   und unabhängig vom Training, also parallel erledigbar.
9. **M3** Golden-Fixtures nicht mehr überschreiben; **M7** Obs-Größenprüfung im Bot;
   **M2** requirements pinnen + `rlbot` ergänzen.
10. **K1** saubere Lösung (Truncation-Flag durch `Gym::StepResult` → `ThreadAgent`), als Patch
    unter `third_party/patches/`. Erst hier, weil es der einzige Upstream-Eingriff ist.

### Stufe 3 — Lernsignal (Verhaltensänderungen, jeweils einzeln, jeweils mit deinem OK)
Ab hier **eine Änderung pro Lauf**, je ~100 Mio. Steps (≈ 25 Minuten) ab demselben Checkpoint und
demselben Seed, verglichen über Ladder + `ep_end_goal` + Entropie:

11. **H2** `ent_coef` 0,01 → 0,004. Erwartung: Entropie fällt unter 3,4, Clip-Fraction > 5 %.
12. **H3** Slot-Shuffle aus. Erwartung: schnellere Verbesserung der gegnerbezogenen Metriken.
13. **K3** Reward-Umgewichtung. Größte erwartete Wirkung, aber auch das längste Neu-Einschwingen —
    deshalb zuletzt, wenn 11 und 12 bereits sauber vermessen sind.
14. Optional: `team_spirit > 0` für zero-sum Shaping. Separater Versuch nach 13.

### Stufe 4 — Geschwindigkeit (erst wenn das Lernsignal stimmt)
15. **H5** `expBufferIterations` konfigurierbar machen, dann A/B über 6 / 3 / 2 Gradientenschritte
    pro Iteration. Bis zu +10 % Durchsatz, aber nur behalten, wenn die Lernkurve nicht leidet.
16. **M6** Obs-Allokationen. Einstelliger Prozentbereich; lohnt nur als Nebeneffekt des
    H3-Umbaus.
17. **N6** AVX512-Zweig verwerfen und das Ergebnis dokumentieren.

### Checkpoint-Kompatibilität

**Keiner** der Punkte in dieser Roadmap ändert die Obs-Größe (257) oder die Aktionszahl (90).
Der Checkpoint bei 2,70 Mrd. Steps bleibt in allen Fällen ladbar.

Einschränkungen, die trotzdem gelten:
* **K3** (Reward-Gewichte) invalidiert den Return-std in `RUNNING_STATS.json` (aktuell 15,12).
  Der Critic muss sich neu einschwingen; rechne mit mehreren zehn Mio. Steps Delle. Der
  Skill-Rating-Verlauf ist über diesen Bruch hinweg nicht vergleichbar.
* **K1** (saubere Lösung) ändert die Value-Targets. Der Critic passt sich an, die Policy bleibt
  gültig.
* **H3** (Shuffle aus) ist unkritisch: Slot 0 war im Training in einem Drittel der Fälle belegt,
  die Policy ist dort in-distribution.
* Würde man dagegen `max_players` oder `action_stack_size` ändern (steht nicht in der Roadmap),
  wäre jeder bestehende Checkpoint unbrauchbar.

---

## 7. Nachtrag aus der Umsetzung (25.09.2026)

Dieser Abschnitt wird während der Umsetzung fortgeschrieben. Der Stand pro Roadmap-Punkt steht in
`AUDIT_PROGRESS.md`.

### 7.1 Widersprüche zwischen Audit und Code

* **H4 überholt:** Das Repository hat inzwischen den Commit `54105bf` (24.09.2026, alles
  versioniert, `rlviser.exe`/`settings.txt` in `.gitignore`). Offen: Tag `baseline-2.7G` und
  Git-Hash in `config_used.json` (beides mit dieser Session erledigt bzw. im Runbook).
* Keine weiteren Widersprüche; alle Zeilenverweise wurden gegen den Code und den gepinnten Upstream
  `ee4cc56` geprüft.

### 7.2 Neuer Befund während der Umsetzung: Bootstrapping an Sammelblock-Grenzen (Upstream)

Beim Bau des K1-Patches fiel eine zweite Ungenauigkeit im Upstream auf, die die Roadmap nicht
enthält. `ThreadAgentManager::CollectTimesteps` hängt die Trajektorien aller 1.024 Spiele
hintereinander (`MultiAppend`). `TorchFuncs::ComputeGAE` benutzt als „nächsten Wert" aber immer
`values[step + 1]`, also den Wert des **nächsten Zustands in der verketteten Liste**. Am Ende jeder
Teil-Trajektorie (markiert als `truncated`) ist das der erste Zustand eines **anderen Spiels**, nicht
der tatsächliche Folgezustand. Nur die allerletzte Trajektorie bekommt über `nextStates[count − 1]`
den richtigen Wert. Betroffen sind ~1.024 von ~100.000 Steps pro Iteration (1 %), der Fehler pro
Ereignis ist die Differenz zweier V-Werte ähnlicher Zustände, also klein — aber es ist genau derselbe
Mechanismus, den K1 für Timeouts braucht. Der K1-Patch behebt beides auf einmal: An jedem
`truncated`-Step wird mit `V(nextStates[step])` gebootstrapt, und `ThreadAgent` legt für beendete
Episoden die **letzte Beobachtung der Episode** (nicht die Reset-Beobachtung) in `nextStates` ab.

### 7.3 Entscheidung zu K3 und `RUNNING_STATS.json` (Return-std 15,12)

**Entscheidung: übernehmen, nicht zurücksetzen.** Begründung:

1. Der Welford-Zähler wächst um höchstens 150 Returns pro Iteration
   (`LearnerConfig::maxReturnsPerStatsInc`), nach 26.789 Iterationen steht er bei rund 4 Millionen
   Samples. Über ein 100-Mio.-Steps-Experiment (~1.000 Iterationen, 150.000 neue Samples) bewegt
   sich der Return-std deshalb praktisch nicht: Die Normierung ist während des gesamten Experiments
   **fest und bekannt** (15,12), egal ob die Reward-Skala passt oder nicht.
2. Ein Reset würde in den ersten Iterationen eine aus wenigen hundert Samples geschätzte, noch
   wandernde Normierung erzeugen — eine zweite Änderung gegenüber der Baseline. Die Vorgabe „jede
   Config ändert genau eine Sache" verbietet das.
3. PPO braucht keine Einheitsvarianz der Returns, nur eine vernünftige Größenordnung: Mit den
   K3-Gewichten liegt ein Tor bei 50 / 15,12 = 3,3 normierten Einheiten (unter der Clip-Grenze 10),
   der dichte Anteil bei ~0,19 / 15,12 ≈ 0,013 pro Step. Der Critic muss sich in beiden Varianten
   neu einschwingen (von ~10 auf ~3 normierte Einheiten Zustandswert); mit „übernehmen" bleiben
   die geloggten Größen (`Avg Val Target`, `Value Function Loss`) in denselben Einheiten wie in der
   Baseline und sind direkt vergleichbar.

Konsequenz für die Abbruchkriterien in `run_experiment.ps1`: Der Value Loss springt bei K3 in den
ersten Iterationen erwartungsgemäß nach oben. Das Kriterium „explodierender Value Loss" wird deshalb
erst nach einer Aufwärmphase (Default 100 Iterationen ≈ 10 Mio. Steps) gegen das Baseline-Niveau
geprüft, siehe `tools/experiments/README.md`.

### 7.4 Entscheidung zu N6 (AVX-512)

Die Messung stammt **lokal vom Ryzen 7 8700F** (`bench/results/config_tuning_avx*.csv`,
`config_tuning_std*.csv`, bestätigt in `docs/phase0_results.md` §7): drei gegen drei Läufe,
68.679 gegen 69.247 SPS, Differenz −0,8 % bei bis zu 7 % Streuung. Gemäß Vorgabe wird der Zweig
**verworfen**; es wird kein Benchmark-Skript dafür gebaut. Die Build-Verzeichnisse und Logs liegen
außerhalb des Repos und werden nicht gelöscht (Vorgabe: keine Dateien löschen); das Runbook nennt
sie als aufräumbar.

### 7.5 K1-Sofortmaßnahme und `sanity.json`

`game_timeout_secs` wird in `lucy_1v1.json` und `lucy_multimode.json` auf 900 gesetzt.
`sanity.json` (120 s, NoTouch 15 s) bleibt unverändert: Es ist der Vier-Minuten-Rauchtest, dessen
Referenzwerte in `docs/phases.md` mit genau dieser Config entstanden sind; der lokale Smoke-Test
vergleicht dagegen.
