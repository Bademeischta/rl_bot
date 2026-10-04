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

**Stand 26.09.2026 (§7.9):** Alle vier gelaufen, dazu eine Wiederholung der Baseline.
Entscheidungen:

* zero_sum **behalten**: +5,8 Tore/Spiel gegen das Baseline-Ende, weit jenseits jedes Rauschens.
* H3 **verlängern**: stärkster Lauf ohne Zero-Sum, aber im Trainingsrauschen. Nächster Test auf
  Zero-Sum-Basis, 3 × 300 Mio. Steps.
* K3 und H2 **verwerfen**.

Vorschlag für den Hauptlauf: `train/configs/lucy_1v1_zero_sum.json`, nicht gestartet.

**Stand 29.09.2026 (§8):** Hauptlauf lief mit `lucy_1v1_zero_sum.json` bis ~6,04 Mrd. Spieltest-Serie:
Anstoß-Drill und (für Teamspiel) 2v2-Anteil mit team_spirit 0,5 behalten, potenzialbasiertes Shaping
und Luftberührungs-Reward verworfen. Vorschläge `lucy_1v1_zero_sum_drill.json` /
`lucy_team_zero_sum.json`, nicht gestartet.

### Stufe 4 — Geschwindigkeit (erst wenn das Lernsignal stimmt)

**Stand 30.09.2026 (§9):** Umgesetzt als G1–G9 (Branch `claude/speed`). Bitgleich und immer an: G2
(+24 %), G3. Als Schalter: G4 (+17 %, bitgleich), Overlap G5, TF32 G6, AMP G7, Stream-Priorität G8.
Vorschlag `lucy_1v1_zero_sum_drill_fast.json` (~2,3–2,7× gegenüber dem bisherigen Hauptlauf),
300-Mio.-Lernvergleich im Trainingsrauschen, gemeinsame Ladder vorn; nicht gestartet. H5 bleibt offen,
M6 nicht nötig (Obs-Aufbau ist kein Engpass).
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
den richtigen Wert. Die Trajektorien sind **pro Spieler** getrennt: Im 1v1 sind das 1.024 Spiele ×
2 Spieler = 2.048 Teil-Trajektorien, also ~2.048 von ~100.000 Steps pro Iteration (~2 %; korrigiert
nach Review R4, vorher stand hier „~1.024 … 1 %"). Der Fehler pro Ereignis ist die Differenz zweier
V-Werte ähnlicher Zustände, also klein — aber es ist genau derselbe Mechanismus, den K1 für
Timeouts braucht. Der K1-Patch behebt beides auf einmal: An jedem `truncated`-Step wird mit
`V(nextStates[step])` gebootstrapt, und `ThreadAgent` legt für beendete Episoden die **letzte
Beobachtung der Episode** (nicht die Reset-Beobachtung) in `nextStates` ab — in der ersten Fassung
des Patches tat er das **nicht** (siehe 7.2b).

### 7.2a Umsetzung K1 (Stand 25.09.2026)

Der Patch `third_party/patches/rlgympppo_cpp_truncation.patch` setzt die im Befund skizzierte
Lösung um, mit zwei Ergänzungen gegenüber der Skizze: (1) `Match::IsDone` wertet **alle**
Bedingungen aus, damit ein Tor im selben Schritt wie ein Timeout als Tor zählt; (2) der
`ThreadAgent` legt für beendete Episoden die letzte Beobachtung der Episode in `nextStates` ab
und die GAE bootstrappt an jedem `truncated`-Step mit `V(nextStates[step])` — ohne (2) würde
ein Timeout mit dem Wert der Reset-Beobachtung der **nächsten** Episode bootstrappen (siehe
7.2). Neue Report-Größe `Truncated Steps` (Timeouts + Blockgrenzen). Die Zählung ist **pro
Spieler**: Jede Spieler-Trajektorie eines Sammelblocks endet mit einem `truncated`-Step (außer sie
endet genau mit einem Tor), im 1v1 mit 1.024 Spielen also **mindestens ~2.048 pro Iteration**, dazu
2 je Timeout (korrigiert nach Review R4; vorher stand hier „~1.024 + Timeouts"). Geprüft in der VM
per C++-Tests und Smoke-Lauf; die Wirkung auf das Lernen ist **lokal** zu messen
(Baseline-Experiment enthält den Patch, Vergleich gegen den alten `metrics.csv`-Verlauf über
`ep_end_time`, `Avg Val Target`).

### 7.2b Korrektur K1b (Review-Befund R4, 25.09.2026, lokal auf dem Trainings-PC)

**Befund:** Die erste Fassung des Patches bootstrappte Timeouts doch von der **Reset-Beobachtung**
der nächsten Episode. `GameInst::Step` hält `auto& nextObs = stepResult.obs` und überschreibt es bei
`done` mit `gym->Reset()`, *bevor* der `ThreadAgent` das `StepResult` sieht; der `ThreadAgent` las
genau dieses `stepResult.obs` als „letzte Beobachtung". Der Unit-Test
`K1_GAE_bootstrappt_Truncation_und_nicht_Terminal` prüfte nur die GAE-Formel mit handgebauten
Eingaben und konnte das nicht bemerken. Folge: Das Ziel am Timeout war `γ·V(Reset-Obs)` statt
`γ·V(letzte Obs)` — besser als 0 (vor K1), aber nicht der richtige Folgewert.

**Korrektur** (`third_party/patches/rlgympppo_cpp_truncation.patch`, neue Fassung):

* `Gym::StepResult::finalObs` (neu): `GameInst::Step` verschiebt (`std::move`, keine Kopie) die
  letzte Beobachtung dorthin, bevor `obs` die Reset-Beobachtung wird — nur an Episodenenden.
* `ThreadAgent` legt für beendete Episoden `finalObs` in `nextStates` ab.
* Diagnose je Iteration in `metrics.csv`: `Timeout Truncations` (Anzahl pro Spieler),
  `Trunc Bootstrap Reset Share` (Anteil der Timeouts, deren Bootstrap-Zustand die Reset-Obs ist:
  muss **0** sein, war vorher 1), `Trunc Bootstrap V Final` / `V Reset` / `V Diff` (mittleres
  V(letzte Obs), V(Reset-Obs) und ihre Differenz in normierten Einheiten wie `Avg Val Target`).
* Schalter `env.timeouts_as_truncation` (Default `true` = korrigiertes Verhalten). `false` meldet
  NoTouch- und Spielzeit-Timeout wieder als echtes Episodenende (Ziel 0 wie vor dem Audit): Rückweg
  und A/B-Vergleich. Die Korrektur an den Blockgrenzen (7.2) bleibt in beiden Stellungen aktiv.
* `Gym.h` definiert `RLGSC_HAS_FINAL_OBS`; das Wurzel-CMake bricht mit einem Upstream-Klon in der
  alten Fassung ab (`tools\apply_patches.ps1 -Reset` setzt ihn zurück und wendet neu an).
* Test über den echten Pfad Gym → GameInst → ThreadAgent → Learner/GAE
  (`tests/cpp/test_truncation.cpp`, `K1b_Echter_Pfad_…`): Bootstrap-Zustand = letzte Beobachtung
  vor dem Reset, ≠ `states[t+1]`, Bootstrap-Wert ≠ `values[t+1]`, Advantage = r + γ·V(letzte Obs) −
  V(s), `Trunc Bootstrap Reset Share` = 0. Mit dem alten Verhalten schlägt er fehl (lokal geprüft).

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

### 7.6 Experiment-Design nach dem Review (25.09.2026, lokal)

* **K3 ist ein Bündel** (Review R11): `k3_rewards.json` ändert **7 Werte** auf einmal (goal,
  concede, `offensive_potential_krc`, `dist_weighted_align_krc`, `velocity_player_to_ball`,
  `save_boost`, `in_air`). Das bleibt so, weil K3 als Gesamtumbau des Reward-Verhältnisses gedacht
  ist; ein Ergebnis lässt sich aber keinem einzelnen Wert zuordnen. `compare.py` liest die
  Änderungen aus den `config.json` der Läufe und markiert jedes Experiment mit mehr als einer
  Änderung als **Bündel** (Tabelle, Abschnitt „Änderungen", Hinweise). Aufgeteilt wird nur, wenn
  das Ergebnis schlecht oder unklar ausfällt.
* **zero_sum statt team_spirit_01** (Review R10): Im 1v1 ist τ wirkungslos (`r_i − r_j`), Zero-Sum
  zählt Tore doppelt, deshalb goal/concede 5. Auch das ist ein Bündel aus 3 Werten, aber mit
  unverändertem Torwert ±10 nach dem Wrapper: gemessen wird nur der Zero-Sum-Effekt auf das
  Shaping. `Average Step/Episode Reward` sind dort konstant 0, `raw_step_reward` zeigt das
  Shaping-Niveau vor dem Wrapper.
* **Hauptkriterium und TrueSkill** (Review R12): Entschieden wird primär über das Duell jedes
  Experiment-Endes gegen das Baseline-Ende (Gewinnrate, Remis = halber Sieg, 95-%-Wilson-KI).
  TrueSkill kommt nur noch aus einer gemeinsamen Ladder, die `compare.py` spielt (alle
  Experiment-Enden, Baseline-Start, Baseline-Ende). Achtung aus dem Mini-Lauf (§7.7): Die meisten
  Duellspiele enden remis; vor Stufe 3 über Spielzahl bzw. `--max-seconds` entscheiden.

### 7.7 Review-Fixes und lokale Verifikation (25.09.2026, Trainings-PC)

Ein unabhängiger Review nach dem Merge von PR #1 fand 18 Befunde (R1–R18); beim Beheben kam
ein Nebenbefund dazu (R19). Status je Befund, Tests und Nachweise stehen in `AUDIT_PROGRESS.md`
(Abschnitt „Review-Fixes"). Die wichtigsten Punkte für dieses Audit:

* **K1b war in der ersten Fassung falsch** und ist korrigiert (§7.2b).
* **Build-Blocker**: Die Patches lagen mit CRLF vor und ließen sich unter Windows nie anwenden
  (R1); alle PowerShell-Skripte brachen unter Windows PowerShell 5.1 an stderr-Ausgaben nativer
  Programme ab (R2) bzw. wurden ohne BOM als ANSI gelesen (R3). Erst danach lief der erste echte
  MSVC-cu128-Build beider Patches.
* **H6 war im 1v1 nicht reproduzierbar** (R19): RocketSim hält die Autos in einem
  `std::unordered_set`, die geseedeten Szenen verteilten ihre Zufallszahlen deshalb
  adressabhängig. Behoben durch Iteration nach Car-ID.
* **Abbruch bei NaN griff nie** (R5): NaN wurde als leeres Feld geschrieben und leere Felder
  wurden ignoriert.

Gemessen auf dem Trainings-PC (lokal / Auditor-Kategorie, §0): Return-std im neuesten Checkpoint
3.907.335.040 = **14,81** (Audit: 15,12 bei 2,70 Mrd.); Kurzläufe `sanity.json` ~74.000 SPS und
`baseline.json` ~71.000 SPS (je nur 2–3 Mio. Steps, keine eingeschwungenen Werte);
Deployment-Latenz p95 0,28 ms. Der Hauptlauf wurde nach dem Audit mit dem alten Binary von
2,70 auf 3,91 Mrd. Steps weitertrainiert; der Checkpoint 2704829056 existiert nicht mehr.

### 7.8 Duell als Messinstrument (26.09.2026, lokal)

Vor Stufe 3 wurde das Duell (`eval/cpp/duel.cpp`) überarbeitet, weil im Probelauf 17 von 20
Spielen remis endeten. Details und Tests in `AUDIT_PROGRESS.md` (Abschnitt Stufe 3).

* **Spiel = 300-s-Match**: nach einem Tor neuer Anstoß bis Spielende (vorher: Ende beim ersten
  Tor oder nach 120 s, Tordifferenz je Spiel nur −1/0/+1). Ergebnis je Spiel im JSON.
* **Unabhängige Spiele**: Seitentausch, Anstöße geseedet (Paare mit Seitentausch teilen die
  Anstoß-Folge), je Spiel frische Arena und eigene Generatoren. Deterministische Policies
  erzeugen Wiederholungen (nur 5 Anstoßpositionen × 2 Seiten) und werden für mehr Spiele
  abgelehnt. Bitgenau reproduzierbar ist ein Spiel wegen RocketSims adressabhängiger Physik
  nicht (31/32 Spiele mit gleichen Toren bei gleichem Seed).
* **Hauptkriterium**: mittlere Tordifferenz pro Spiel mit 95-%-t-Intervall; dazu Gewinnrate
  (Wilson) und Tore pro Minute.

Messungen mit dem neuesten Checkpoint 3.907.335.040 (je 1000 Spiele à 300 s, Seed 123):

| Duell | Tordifferenz/Spiel [95-%-KI] | SD | Gewinnrate [KI] | Tore/min |
|---|---|---|---|---|
| **Nullmessung** 3,907 gegen sich selbst (Endstand-Binary, 8 Threads) | −0,005 [−0,058; +0,048] | 0,854 | 49,8 % [46,8; 52,9] | 0,144 |
| Nullmessung, erste Fassung (1 Thread, ohne frische Arena) | −0,048 [−0,101; +0,005] | 0,859 | 46,9 % [43,8; 49,9] | 0,150 |
| 3,907 gegen 3,807 (100 Mio. Steps älter) | **+0,310 [+0,250; +0,370]** | 0,969 | 59,4 % [56,3; 62,4] | 0,172 |
| 3,907 gegen 3,682 (225 Mio. Steps älter) | **−0,111 [−0,181; −0,041]** | 1,129 | 46,7 % [43,6; 49,8] | 0,239 |

Folgerungen:

* Die Nullmessung liegt um 0, die Intervallbreite ±0,053 Tore/Spiel ist das Rauschen bei 1000
  Spielen. (Die Gewinnrate der ersten Nullmessung schloss 50 % knapp aus: ein erwartbarer
  5-%-Ausreißer und ein weiterer Grund, sie nicht als Hauptkriterium zu nehmen.)
* **Spielanzahl 1000** (`run_experiment.ps1 -DuelGames`): Realistische Effekte zwischen
  Checkpoints liegen hier bei 0,1–0,3 Tore/Spiel; 1000 Spiele lösen ±0,053 (Null-SD) bis ±0,07
  (SD der Effekt-Duelle) auf, also Effekte ab ~0,1 Tore/Spiel mit hoher Wahrscheinlichkeit. Das
  kostet ~6 min je Duell (8 Threads, 0,35 s je Spiel; ein Kern: 1,14 s). 100 Spiele (vorher)
  hätten ±0,17 aufgelöst, also fast nichts.
* **Spielstärke im Selbstspiel ist nicht transitiv**: Der neueste Checkpoint schlägt den 100
  Mio. Steps älteren deutlich und verliert gegen den 225 Mio. Steps älteren. Ein Duell gegen nur
  **einen** Gegner (Baseline-Ende) kann deshalb eine Stil-Frage messen (wer kontert wen) statt
  allgemeiner Stärke. Die gemeinsame Ladder und mehrere Gegner gehören in jede Entscheidung.

### 7.9 Ergebnisse Stufe 3 (26.09.2026, lokal)

**Aufbau.** Die Experimente liefen in der Reihenfolge des Runbooks (Baseline, H2, H3, K3,
zero_sum), dazu kam als sechster Lauf eine **Wiederholung der Baseline**. Jeder Lauf: 100 Mio.
Steps ab 3.907.335.040, Seed 123, Build `d7c4b0a` (`_git` in jeder `config_used.json` geprüft).
Die Läufe liefen nacheinander, je ~39 min (25 min Training bei 68.300–69.700 SPS, dazu zwei
Duelle à ~6,3 min; die Baseline hat nur eines, ~33 min). **Kein Abbruchkriterium hat gegriffen.** Bewertet wird jedes Ende gegen vier
Gegner, je Duell 1000 Spiele bzw. 500 im Panel, Spiele à 300 s:

* den Start
* das Baseline-Ende
* zwei ältere Hauptlauf-Checkpoints, 3,682 und 3,807 Mrd. (Panel wegen der Nicht-Transitivität, §7.8)

Dazu kommt die gemeinsame Ladder aus `compare.py` (100 Spiele je Paarung). Sie lief zweimal mit
gleicher Reihenfolge, μ−3σ wich um höchstens 0,3 ab. Rohdaten:
`results\exp_*_2026-09-26_*`, `results\stage3_panel\`, `results\compare_stage3*.md` (lokal,
`results\` ist nicht versioniert).

Tordifferenz pro Spiel aus Sicht des Experiment-Endes, [95-%-KI]:

| Ende | gg. Start | gg. Baseline-Ende | gg. 3,682 | gg. 3,807 | Mittel über Start, 3,682, 3,807 (Abstand zum Mittel beider Baseline-Läufe) | Ladder μ−3σ |
|---|---|---|---|---|---|---|
| baseline | −0,00 [−0,03; +0,03] | (Referenz) | −0,17 [−0,23; −0,11] | −0,08 [−0,13; −0,02] | −0,08 (−0,16) | 19,6 |
| Wiederholung der Baseline | +0,24 [+0,20; +0,27] | +0,08 [+0,06; +0,11] | +0,17 [+0,12; +0,22] | +0,31 [+0,25; +0,37] | +0,24 (+0,16) | 25,9 |
| H2 `ent_coef` 0,004 | −0,03 [−0,06; +0,01] | −0,03 [−0,04; −0,01] | −0,29 [−0,36; −0,23] | −0,13 [−0,19; −0,07] | −0,15 (−0,23) | 17,7 |
| H3 ohne Slot-Shuffle | +0,63 [+0,57; +0,69] | +0,41 [+0,36; +0,46] | +0,53 [+0,45; +0,62] | +0,68 [+0,59; +0,76] | +0,61 (+0,53) | 31,0 |
| K3 (Bündel, 7 Werte) | +0,39 [+0,33; +0,44] | +0,32 [+0,27; +0,37] | +0,37 [+0,28; +0,47] | +0,69 [+0,59; +0,80] | +0,49 (+0,41) | 25,7 |
| zero_sum (Bündel, 3 Werte) | **+6,79** [+6,61; +6,97] | **+5,76** [+5,58; +5,93] | **+6,21** [+5,97; +6,45] | **+7,24** [+6,97; +7,51] | **+6,75 (+6,67)** | 38,7 |

Trainingsmetriken, letztes Fünftel der Iterationen:

| Ende | Entropie (Start 3,58) | Clip-Frac. | KL | `ep_end_goal` | Zeit-Timeout-Anteil | Value Loss | Ballkontakt | Step-Reward (roh) |
|---|---|---|---|---|---|---|---|---|
| baseline | 3,43 | 2,21 % | 0,0025 | 0,39 | 0,61 | 0,066 | 0,036 | 0,98 |
| Wiederholung | 3,40 | 2,20 % | 0,0025 | 0,38 | 0,61 | 0,068 | 0,034 | 0,99 |
| H2 | 2,68 | 1,85 % | 0,0023 | 0,11 | 0,89 | 0,008 | 0,042 | 1,22 |
| H3 | 3,44 | 2,17 % | 0,0025 | 0,36 | 0,64 | 0,068 | 0,037 | 0,99 |
| K3 | 3,99 | 0,80 % | 0,0012 | 0,39 | 0,61 | 0,005 | 0,028 | 0,21 |
| zero_sum | 2,89 | 3,00 % | 0,0031 | 1,00 | 0,00 | 0,370 | 0,051 | 0,72 |

**Rauschen zwischen Trainingsläufen.** Das ist die wichtigste methodische Erkenntnis dieser
Stufe. Die Wiederholung der Baseline hat dieselbe Config, denselben Start, Seed und Build. Die
Läufe laufen nur über RocketSims nicht bitgenaue Physik auseinander. Ihre Trainingsmetriken sind
fast identisch. Im Duell ist sie aber gegen jeden gemeinsamen Gegner besser als der erste
Baseline-Lauf: +0,24, +0,34 und +0,39 Tore/Spiel, im Mittel 0,32. Das ist das Sechsfache der
Duell-Rauschbreite (±0,053, §7.8). Aus diesem einen Paar geschätzt streut ein einzelner
100-Mio.-Lauf um σ ≈ 0,23 Tore/Spiel. Der Abstand eines einzelnen Experiment-Laufs zum Mittel
beider Baseline-Läufe streut damit um σ ≈ 0,28 (σ·√1,5). Das ist der Maßstab in den
Entscheidungen unten. Weil der Seed gleich war, ist das eher eine Untergrenze. Folgen:

* Das Duell-KI eines einzelnen Laufs sagt nichts über dieses Rauschen. Die Wiederholung hätte
  nach dem alten Hauptkriterium als „besser“ gegolten (+0,083, KI ganz über 0).
* `compare.py` erkennt Wiederholungen deshalb jetzt und nennt Effekte bis zum Doppelten ihres
  Abstands „im Trainingsrauschen“ (D4 in `AUDIT_PROGRESS.md`). Das Runbook verlangt zu jeder
  Reihe eine Wiederholung.
* Ein 100-Mio.-Lauf pro Config löst nur Effekte ab etwa 0,5 Tore/Spiel sicher auf.

**Entscheidungen**

* **zero_sum: behalten.** Der Effekt ist über zwanzigmal so groß wie das Trainingsrauschen und
  zeigt sich gegen alle vier Gegner gleich: 99,4–99,8 % Gewinnrate, in 3000 Spielen **kein
  einziges verloren**. In der Ladder liegt zero_sum mit Abstand vorn. Die Trainingsmetriken passen
  dazu: Fast jede Episode endet mit einem Tor (`ep_end_goal` 1,00 statt 0,39), die
  900-s-Timeouts verschwinden (0,61 → 0,00), der Ballkontakt steigt um 39 %. Damit ist Kurzfazit
  Punkt 1 bestätigt. Ohne Zero-Sum zahlt das dichte Shaping beiden Selbstspiel-Agenten zugleich.
  Das Ergebnis ist ein passives Gleichgewicht: 61 % der Baseline-Episoden enden am Zeitlimit, im
  Duell fallen 0,02–0,07 Tore/min. Mit `r_i − r_j` lohnt nur noch, was dem Gegner schadet. Kosten
  und Beobachtungspunkte:
  * Value Loss 0,37 statt 0,066 (der Wert hängt jetzt vom Spielstand ab).
  * `Avg Val Target` fällt noch (3,5 → 2,4; bei Zero-Sum ist im Mittel 0 zu erwarten).
  * Die Entropie sinkt auf 2,89. Warnschwelle ist 2,5.
  * Vorbehalt: Die Höhe des Vorsprungs spiegelt auch die Passivität aller Gegner. Gegen einen
    Zero-Sum-Gegner wird er viel kleiner ausfallen.
* **H3 (Slot-Shuffle aus): verlängern.** H3 ist gegen alle vier Gegner der stärkste Lauf ohne
  Zero-Sum (+0,41 bis +0,68) und in der Ladder Zweiter. Dafür gibt es einen plausiblen Grund: Duell
  und Bot (`env/obs_python.py`) mischen nicht, der Gegner steht immer in Slot 0, und H3 trainiert
  genau so. Die Trainingsmetriken sind dabei unverändert. Der Vorsprung auf die bessere der beiden
  Baseline-Läufe (+0,37 im Mittel) ist aber so groß wie der Abstand der Baseline-Läufe
  untereinander (0,32). Gegen das Mittel beider Baseline-Läufe sind es +0,53 ≈ 1,9σ, und σ
  stammt aus einem einzigen Paar. Das liegt **im Trainingsrauschen** und ist nicht bestätigt.
* **K3 (Reward-Umgewichtung, Bündel): verwerfen in dieser Form.** Das Ergebnis liegt im
  Trainingsrauschen: +0,41 gegen das Mittel der Baseline-Läufe ≈ 1,5σ, und in der Ladder liegt
  K3 gleichauf mit der Wiederholung der Baseline (25,7 gegen 25,9). Dazu kommen Nebenwirkungen:
  * Der Trainer teilt die Rewards durch die übernommene Return-std (14,81) und normiert
    Advantages nicht. Mit fünfmal kleinerem Shaping werden die Policy-Gradienten deshalb kleiner:
    Die Entropie **steigt** (3,58 → 3,99), die Clip-Fraction fällt auf 0,8 %, die KL halbiert sich.
  * Der erhoffte Anstieg von `ep_end_goal` bleibt aus (0,39).
  * Der Ballkontakt fällt um 24 %.

  Das Ziel von K3 (Tore zählen mehr als Shaping) erreicht zero_sum direkter. Ein neuer Versuch
  wäre nur auf Zero-Sum-Basis sinnvoll, mit neu geschätzter Return-std.
* **H2 (`ent_coef` 0,004): verwerfen.** Im Duell liegt H2 gegen alle vier Gegner unter der
  Baseline (−0,03 bis −0,29), allerdings noch im Trainingsrauschen (−0,23 ≈ −0,8σ). Eindeutig
  sind die Trainingsmetriken. Die Entropie fällt (2,68), aber die Updates werden nicht größer:
  Clip-Fraction 1,85 %, KL 0,0023 statt der erwarteten > 5 % bzw. 0,006. Stattdessen sammelt die
  Policy mehr Shaping (Step-Reward +25 %) und schießt weniger Tore (`ep_end_goal` 0,11,
  Zeit-Timeouts 89 %). H2 vertieft die Passivität.

**Verlängerung (Vorschlag, nicht gestartet).** Die einzige offene Frage mit Aussicht ist H3. Der
Hauptlauf soll mit Zero-Sum weitergehen, deshalb wird H3 auf dieser Basis geprüft
(`experiments/zero_sum_h3.json`, genau eine Änderung gegenüber `zero_sum.json`). Geplant sind drei
Läufe à 300 Mio. Steps vom selben Start (Befehle in `LOCAL_RUNBOOK.md` §5a):

* zero_sum (Referenz)
* zero_sum_h3
* eine Wiederholung von zero_sum (Trainingsrauschen)

Jeder Lauf dauert ~90 min (~73 min Training bei ~69.000 SPS, zwei Duelle, Lauf-Ladder),
zusammen **~4,5 h**. Belastbarer wären zwei Läufe je Config (~9 h). Unter Zero-Sum fallen viele
Tore, und die Streuung je Spiel liegt bei SD ≈ 2,8–3,0 statt 0,85. 1000 Duellspiele lösen dann
±0,18 Tore/Spiel auf; das reicht bei Spielständen von 6–7 Toren pro Spiel. K3 und H2 nicht
verlängern.

**Config für den fortgesetzten Hauptlauf (Vorschlag, nicht gestartet):**
`train/configs/lucy_1v1_zero_sum.json` = `lucy_1v1.json` plus die drei zero_sum-Werte
(`team_spirit` 0,1, `goal`/`concede` 5). Obs 257, 90 Aktionen, `max_players` und
`action_stack_size` bleiben unverändert, der Checkpoint ist kompatibel. `RUNNING_STATS.json` wird
wie im Experiment übernommen. H3 kommt erst nach bestätigter Verlängerung dazu. Beim Fortsetzen
rotiert `checkpoints_to_keep` 10 die ältesten Checkpoints aus `runs\lucy_1v1` heraus. Wer 3,682 bis
3,907 Mrd. als Duell-Gegner behalten will, sichert sie vorher.

## 8. Spieltest im echten Rocket League (28.09.2026, lokal)

### 8.1 Auftrag und Ausgangslage

Der Hauptlauf lief mit `train/configs/lucy_1v1_zero_sum.json` von 3,907 bis ~6,04 Mrd. Steps
(neuester Checkpoint 6.037.692.544, gestoppt). Spieltest mit RLBot v5 offline: Der Bot schlägt den
Psyonix-Allstar, verliert gegen einen Menschen vor allem über die Anstöße, fährt vor dem gegnerischen
Tor lange von Ecke zu Ecke, spielt kaum Luftbälle, und zwei Instanzen im selben 2v2-Team jagen beide
dem Ball hinterher. Auftrag: Ursachen finden und mit Experimenten belegen; Hauptlauf nicht anfassen.

Vorgehen: Phase A baut Mess- und Experimentierwerkzeuge (Details und Commits in `AUDIT_PROGRESS.md`,
Abschnitt Spieltest), Phase B misst ohne Training am Checkpoint 6.037.692.544, Phase C sind die
Trainings-Experimente (nur nach OK des Nutzers). Rohdaten Phase B: `results\phase_b_2026-09-28\`
(`report_phase_b.md`, lokal, `results\` ist nicht versioniert).

### 8.2 Messwerkzeuge (Phase A)

* `env/cpp/PlayStats` (PlayTracker): Anstoß (erste Berührung tickgenau, Tempo, Boost, Ballhälfte und
  Ballnähe 3 s danach, Tor in 10 s), Aufenthalte im Angriffsdrittel (mit/ohne Tor, ab 8 s „lang“),
  Schüsse aufs Tor (Kontakt, nach dem der Ball laut Wurfparabel in 3 s ins Tor fliegt), Ballkontakte
  mit Höhe am Boden/in der Luft (Aerial: Ball ab 450 uu), im Teamspiel Mitspielerabstand,
  Double-Commit (zwei Mitspieler näher als 1500 uu und mit über 500 uu/s auf den Ball zu) und
  Absicherung (mindestens einer 500 uu hinter dem Ball). Als Trainingsmetriken in `metrics.csv`
  (2v2/3v3 mit Präfix) und je Seite im Duell (`duel.exe`, `"stats"`).
* Im Selbstspiel ist „wer gewinnt den Anstoß“ konstruktionsbedingt 50 %. Die Trainingsmetriken
  messen deshalb Zeit bis zur ersten Berührung, Tempo, Boost und Anstoß-Tore; die Gewinnrate gibt es
  im Duell und in `eval/kickoff_eval.py` (gegen andere Checkpoints oder geskriptete Anstöße).
* `duel.exe --a-mode/--b-mode sample|argmax|argmax_group`; `eval/kickoff_eval.py` (Bot-Code im
  Python-RocketSim, Schrittfolge wie Gym::Step, ein Test prüft den Snapshot-Zeitpunkt);
  `tools/cpp/reward_budget.exe` (Reward je Komponente über den echten Code, Summe gleich dem
  Match-Reward bis auf 2e-6).

### 8.3 Aktionsauswahl: Der Bot spielt argmax, das Training zieht

`deploy/rlbot/bot.py` wählt die wahrscheinlichste Aktion, Training und Duell ziehen aus der
Verteilung. Die Aktionstabelle hat am Boden 9 gleichwirkende Einträge für „Gas + Boost“, aber nur
einen für „Gas ohne Boost“: argmax verteilt nicht nach Wirkung und boostet seltener, als die Policy
will (in einer Selbstspiel-Stichprobe boostet argmax in 24,5 % der Lagen mit Boost-Wahrscheinlichkeit
über 50 % nicht; 36 % der Bodenentscheidungen weichen von der wahrscheinlichsten Wirkung ab).

Gemessen (Checkpoint gegen sich selbst, 1000 Spiele à 300 s, B zieht):

| A | Tordifferenz/Spiel [95-%-KI] | Siege A:B | Ballkontakte/min A:B | Anstoß zuerst A |
|---|---|---|---|---|
| Ziehen (Nullmessung) | −0,05 [−0,15; +0,05] | 364:391 | 29,3 : 29,3 | 50 % |
| **argmax (Bot)** | **+1,21** [+1,11; +1,32] | 669:131 | 41,2 : 23,9 | 64 % |
| argmax über Wirkungsklassen | +0,69 [+0,58; +0,80] | 529:252 | 36,6 : 25,5 | 77 % |

Folgerung: Der Bot spielt im Spiel bereits im stärksten der drei Modi. Die Boost-Verzerrung schadet
nicht messbar (vermutlich spart sie Boost); argmax über Wirkungsklassen gewinnt mehr Anstöße, spielt
insgesamt aber schwächer. **Keine Änderung an `bot.py`.** Für die Experimente heißt das: Das Duell
(Ziehen gegen Ziehen) misst nicht genau das Spiel des Bots; Anstoß-Auswertung und Kennzahlen je
Seite gibt es deshalb zusätzlich mit argmax.

### 8.4 Anstoß

Was der Bot tut (`eval/kickoff_eval.py --trajectory`, gegen sich selbst, argmax wie im Spiel):

| Position | erste Berührung | Tempo des Berührers | Spitze | Sprung | Flip | Boost |
|---|---|---|---|---|---|---|
| diagonal links | 2,57 s | 763 uu/s | 1961 | 2,34 s | – | 45 |
| diagonal rechts | 3,09 s | 1124 | 1393 | 2,74 s | 3,27 s | **0** |
| versetzt links | 2,79 s | 1378 | 1816 | 2,61 s | 2,88 s | 40 |
| versetzt rechts | 2,79 s | 1296 | 1648 | 2,54 s | – | 37 |
| Mitte | 3,01 s | 1323 | 1994 | 2,74 s | – | 46 |
| *Speedflip-Skript* | *1,89 / 2,14 / 2,47 s* | *2300* | *2300* | *0,44–0,51 s* | *0,50–0,56 s* | |

Der Bot erreicht nie Höchsttempo, nimmt etwa 1 s vor dem Ball Gas weg, springt 0,15–0,35 s vor der
Berührung und flippt, wenn überhaupt, erst danach. Diagonal rechts boostet er gar nicht (die Policy
selbst will dort nur zu 23 % boosten: eine Links/Rechts-Asymmetrie). Er kommt 0,5–0,9 s nach einem
Speedflip an, mit 800–1400 statt 2300 uu/s. Im Selbstspiel ist das ein Gleichgewicht (beide gleich
langsam, 50 %), gegen einen schnellen Menschen verliert er jeden Anstoß.

Gegen geskriptete Anstöße (`deploy/scripted_kickoff.py`, nach der ersten Berührung spielt die
Policy weiter; je 1000 Anstöße, alle Positionen, Seiten gewechselt):

| A (Skript, danach Policy) | B | A zuerst | Ball in Bs Hälfte nach 3 s | Tore in 10 s A:B |
|---|---|---|---|---|
| Speedflip | Policy argmax (Bot) | 100 % | 90 % | **500:0** |
| Speedflip | Policy zieht | 100 % | 99,7 % | 575:0 |
| Frontflip | Policy argmax | 100 % | 65 % | 0:0 |
| Frontflip | Policy zieht | 100 % | 64 % | 145:9 |
| Policy zieht | Policy zieht | 50 % | 52 % | 14:11 |

Ursachen: (1) Anstöße sind im Training selten. Jede Episode endet mit einem Tor, nur 2 von 7 neuen
Episoden beginnen mit Anstoß; bei ~107 s Episodenlänge sind ~0,8 % der Trainingsdaten Anstoßphase
(im echten Spiel eher 5 %). (2) Im Selbstspiel gibt es keinen Druck, schneller als ein Gegner zu
sein, der selbst nie schnell ist. (3) Nicht die Ursache: Timing oder Deployment (Nutzer: pünktlich bei
„GO“; die Beobachtung beim Anstoß ist wie im Training, der Paketpuffer wird beim Anstoß geleert).

**Geskripteter Anstoß im Deployment (Bewertung; auf Wunsch des Nutzers als Option eingebaut, Standard aus):**
* Nutzen: sofort und groß. In RocketSim gewinnt der Speedflip jede erste Berührung gegen die
  aktuelle Policy; gegen einen Menschen mindestens Gleichstand.
* Aufwand: mittel. Das Skript existiert und ist getestet, es braucht nur, was RLBot liefert. Einbau
  in `bot.py` hieße: Anstoß erkennen (Phase Kickoff, Ball in der Mitte), pro Tick steuern, nach der
  ersten Berührung (spätestens nach 3 s) an die Policy übergeben, Aktionshistorie mit den nächsten
  Tabelleneinträgen füllen, dazu Tests.
* Risiken: Abweichung zwischen RocketSim und Spiel beim Flip-Timing (RocketSim ist nah am Spiel,
  Speedflips aus RLBot-Bots funktionieren dort; verpasste Pakete kosten Ticks), ein Gegner, der den
  Anstoß antäuscht (das Skript fährt stur), 2v2 (wer fährt, wer bleibt hinten), und die Policy lernt
  den Anstoß nie selbst.
* Einschätzung: Für Spiele gegen Menschen der schnellste Hebel. Der gelernte Weg (Phase C, K1/K2)
  bleibt sinnvoll, wird aber in 300 Mio. Steps kaum 0,5–0,9 s aufholen.
* Umgesetzt (KO4, 28.09.2026): `$env:RLBOT_SCRIPTED_KICKOFF = "speedflip"` vor dem Start von RLBot.
  Das Skript steuert pro Tick, solange die Phase „Kickoff“ ist und der Ball ruhend in der Mitte liegt
  (höchstens 4 s), im Teamspiel nur beim ballnächsten Mitspieler; danach entscheidet sofort die
  Policy, mit den Skript-Eingaben als nächste Tabelleneinträge im Aktions-Stack. Test in RocketSim
  über echte RLBot-Pakete: erste Berührung diagonal unter 2,0 s. Im echten Spiel ungeprüft.

### 8.5 Ecken-Schleife vor dem gegnerischen Tor

Reward-Bilanz über den echten Code (`reward_budget.exe`, Config des Hauptlaufs, Zero-Sum,
γ = 0,9954, Horizont 14,5 s). Ein Tor ist nach Zero-Sum +10 wert und beendet die Episode.

| Lage | Shaping A − D pro Sekunde | ein Tor entspricht |
|---|---|---|
| Ball in der Ecke, Verteidiger steht im Tor (feste Lage) | +17,3 | 0,58 s dieser Lage |
| dasselbe, Verteidiger greift an | +21,7 | 0,46 s |
| quer an der Grundlinie | +6,5 | 1,5 s |
| Schussposition frontal | +13,9 | 0,72 s |
| gespielt (argmax, 300 Spiele): Ball im Angriffsdrittel, Ecke (44 % der Zeit) | +2,4 | 4,1 s |
| gespielt: Angriffsdrittel Mitte (23 %) | +2,0 | 4,9 s |
| gespielt: Mittelfeld (33 %) | 0,0 | – |

Fast alles kommt aus `offensive_potential_krc` und `dist_weighted_align_krc`;
`touch_ball_to_goal_accel` bringt für einen harten Schuss einmalig ~0,4. Gegen einen Gegner, der
nicht angreift (Mensch im Tor), zahlt die Lage „Ball in der Ecke, ich dahinter“ 17–22 pro Sekunde;
drei Sekunden davon sind diskontiert so viel wert wie fünf Tore. Im Selbstspiel greift der Gegner an
und beendet die Lage schnell, deshalb fallen dort trotzdem Tore.

Im Duell (argmax gegen Ziehen, je Seite): Der Ball ist 44 % der Zeit im Angriffsdrittel des
argmax-Bots, **53 % seiner Aufenthalte dort dauern 8 s oder länger**, 23 s pro Spielminute ohne Tor;
17 % der Aufenthalte enden mit Tor. Das passt zum Spieltest. Hebel: Shaping, das Halten nicht
bezahlt (potenzialbasiert, Phase C E1), oder ein höherer Torwert (E2).

### 8.6 Luftspiel

Der Bot hat praktisch keine Aerials: 0,01 Ballkontakte pro Spielerminute mit Ball über 450 uu
(argmax, Duell). Die „Luftkontakte“ (5–6 pro Minute) sind Sprünge an einen Ball in ~150 uu Höhe.
Der Bruch im Hauptlauf ab ~5,6 Mrd. Steps (in_air_ratio 0,065 → 0,12, Ballkontakt 0,050 → 0,033) ist
genau das: Der Bot springt in Zweikämpfe (Luftkontakte 0,15 → 5,4 pro Minute). Das hat ihn stärker
gemacht: 6,04 schlägt 5,56 Mrd. mit +0,55 [+0,43; +0,66] Toren pro Spiel (1000 Spiele). `in_air`
(0,02/Step) zahlt jedes In-der-Luft-sein, belohnt aber keinen Kontakt; die Aerial-Szene hat 0,5 von
7 Gewichtsanteilen.

### 8.7 Teamspiel

* **Nur 1v1 trainiert, bestätigt:** `mode_mix` [1,0,0] in `config_used.json`, und die Gewichte der
  ersten Schicht für die beiden Mitspieler-Slots (Eingänge 112–169) sind zwischen 5,56 und 6,04 Mrd.
  bitgleich und haben exakt die Statistik der Initialisierung (|W| 0,0313, erwartet 0,0312; trainierte
  Spalten 4–8× größer): Sie haben nie einen Gradienten bekommen. Im 2v2 speist der Mitspieler Rauschen
  über Zufallsgewichte ein, und die Policy sieht erstmals zwei Gegner gleichzeitig.
* **Bereit ohne Obs-/Aktionsänderung:** Padding für 3 Spieler, C++/Python-Parität für 2v2/3v3
  (Golden-Fixtures, je 10 Fälle), Modus-Mix je Env, `duel.exe --team-size`, zwei Bot-Instanzen im
  selben Team (neuer Test: jede baut die Obs aus ihrer Sicht, Mitspieler im Mitspieler-Slot).
* **Nicht bereit:** `lucy_multimode.json` ist vom 1v1-Checkpoint nicht ladbar (Netz 1024/1024/512/512)
  und zählt Tore doppelt (goal 10 mit team_spirit 0,2). Für Phase C gibt es `sp_mode_2v2.json`.
* **Ist-Zustand 2v2** (6,04 gegen sich selbst, 500 Spiele): Double-Commit 10,8 % der Team-Zeit,
  Absicherung 68 %, Mitspielerabstand 1611 uu, nur 7,8 Ballkontakte je Spielerminute (1v1: 29).
* **team_spirit τ:** ZeroSumReward gibt jedem (1 − τ/2) seines eigenen Shapings plus τ/2 des
  Mitspielers, minus das Mittel der Gegner. Bei 0,1 sind das 95 % Eigenanteil. Auch τ = 1 lässt 50 %,
  weil alle Shaping-Terme „ich nah am Ball“ belohnen: Der Anreiz, dass beide hinfahren, halbiert sich
  nur. Vorschlag: τ = 0,5 beim Einstieg in 2v2 (Zuordnung von Belohnung zu Aktion bleibt klar,
  Egoismus halbiert), nach 300–500 Mio. Steps mit 2v2-Anteil und stabilen Team-Kennzahlen 1,0
  (Config-Wechsel, der Checkpoint bleibt kompatibel). Gegen Double-Commits wirkt erst ein
  Team-Shaping, bei dem nur der ballnächste Mitspieler die Ballnähe-Terme bekommt (nicht gebaut,
  möglicher Folgeversuch).

### 8.8 Plan Phase C (vom Nutzer freigegeben: Kernserie 1, 2, 3, 5, 7, 10, 11)

Je 300 Mio. Steps ab 6.037.692.544, Seed 123, nacheinander, Referenz `experiments/zero_sum.json`
(= Hauptlauf-Config als Experiment) mit Wiederholung. Jede Config ändert genau eine Sache
(`tests/test_experiment_configs.py`).

| # | Config | Änderung |
|---|---|---|
| 1, 2 | `zero_sum` und Wiederholung | – (Trainingsrauschen bei 300 Mio.) |
| 3 | `sp_kickoff_drill` | Szene Anstoß-Drill, Gewicht 4, nach 6 s abgeschnitten |
| 4 | `sp_kickoff_first_touch` | +2 für die erste Berührung nach dem Anstoß |
| 5 | `sp_potential_shaping` | Offensiv-Shaping als Potenzialdifferenz, Faktor 8 (kalibriert) |
| 6 | `sp_goal_x3` | Torwert ±30 statt ±10 |
| 7 | `sp_air_touch` | Luftberührung, Gewicht 3, skaliert mit der Höhe |
| 8 | `sp_aerial_share` | Aerial-Szene 0,5 → 2,0 |
| 9 | `sp_no_in_air` | in_air 0,02 → 0 |
| 10 | `sp_mode_2v2` | mode_mix [3,1,0], mit `-TeamDuel` |
| 11 | `sp_mode_2v2_tau05` | wie 10, team_spirit 0,5 (Vergleich gegen 10) |

Laufzeit je Lauf ~95 min (~71 min Training bei ~70.000 SPS, zwei Duelle, zwei Anstoß-Auswertungen,
Lauf-Ladder), Teamläufe ~115 min. Kernserie (1, 2, 3, 5, 7, 10, 11) ~11,5 h, alle elf ~18 h.
Begründung für 300 statt 100 Mio. Steps: Zwei gleiche 100-Mio.-Läufe lagen in Stufe 3 um ~0,3
Tore/Spiel auseinander; die gesuchten Fähigkeiten sind seltene Ereignisse (in 100 Mio. Steps ~9000
Anstöße); der Bruch bei 5,6 Mrd. zeigt, dass sich Verhalten in ~100 Mio. Steps verschieben kann,
300 Mio. geben Luft. Effekte unter dem Doppelten des Abstands der beiden Referenzläufe gelten als
„im Trainingsrauschen“ und werden verlängert statt entschieden.

### 8.9 Ergebnisse Phase C, Kernserie (28./29.09.2026)

Vom Nutzer freigegeben: Kernserie, sieben Läufe à 300 Mio. Steps ab 6.037.692.544, Seed 123, Build
`7d144df` (sauber, `run_all_checks.ps1` davor komplett grün). Alle sieben ohne Abbruch, 21:32 bis
09:25. Während der ersten Läufe lief auf dem PC ein Spiel (VALORANT); das bremste den Durchsatz
(46.700 statt ~70.000 SPS im Referenzlauf), verändert aber keine Lernergebnisse (die hängen an den
Steps). Die SPS-Frage ist deshalb separat gemessen (unten). Rohdaten:
`results\exp_{zero_sum,replicate_zero_sum,sp_*}_2026-09-2*`, Vergleich
`results\phase_c_2026-09-28\compare_kern.md`, `compare_team.md`, `joint_ladder.json`.

**Trainingsrauschen bei 300 Mio. Steps ist groß.** Zwei Läufe derselben Config (`zero_sum` und
Wiederholung) liegen im Duell gegen den gemeinsamen Start bei −3,24 und +0,19 Toren/Spiel. Der
Referenzlauf ist ab etwa der Hälfte in einen langsamen Anstoß abgedriftet (Zeit bis zur ersten
Berührung im Training 2,9 → 3,6 s, Tore/min fallen); sein Ende verliert gegen den Start 509 von 1000
Anstößen direkt mit Gegentor in 10 s. Das ist Selbstspiel-Drift: Beide Seiten werden gemeinsam
langsam, ein Gegner von außen nutzt das aus. Folgen für die Auswertung: Das Ende des Referenzlaufs
taugt allein nicht als Referenz (compare.py nennt das Rauschen 3,4 Tore/Spiel, damit wäre jedes
Ergebnis „im Rauschen“). Bewertet wird deshalb gegen den gemeinsamen Start-Checkpoint, gegen
**beide** Referenzläufe und über die Kennzahlen, auf die das Experiment zielt.

| Lauf | Duell gegen Start: Tordifferenz/Spiel [95-%-KI] | Ladder μ−3σ | Anstöße gegen Start: A zuerst, Tore in 10 s | Anstoß-Zeit im Training | Aerials/Spielermin. | Drittel: Konversion / lang |
|---|---|---|---|---|---|---|
| Start 6,04 Mrd. | – | 26,29 | – | 2,9 s (Anfang der Läufe) | 0,013 | – |
| zero_sum (Referenz) | −3,24 [−3,44; −3,04] | 24,33 | 0 %, 7 : 509 | 3,45 s | 0,013 | 0,136 / 0,433 |
| Wiederholung | +0,19 [+0,07; +0,32] | 26,77 | 4 %, 17 : 128 | 3,10 s | 0,012 | 0,201 / 0,422 |
| sp_kickoff_drill | +0,28 [+0,18; +0,39] | **27,39** | **23 %, 30 : 13** | **3,00 s** | 0,018 | 0,124 / 0,423 |
| sp_potential_shaping | **−14,15** [−14,35; −13,95] | 1,62 | 0 % | 6,61 s | 0,000 | – (kaum Ballkontakt) |
| sp_air_touch | +0,74 [+0,62; +0,87] | 26,97 | 27 %, 52 : 35 | 3,59 s | **0,010** | 0,237 / 0,408 |
| sp_mode_2v2 | +0,39 [+0,27; +0,50] | 26,40 | 25 %, 41 : 10 | 2,84 s | 0,012 | 0,190 / 0,387 |
| sp_mode_2v2_tau05 | +0,43 [+0,31; +0,55] | – | 40 %, 55 : 13 | 2,86 s | 0,014 | 0,189 / 0,400 |

(„Drittel lang“ = Anteil der Aufenthalte im Angriffsdrittel ab 8 s. Tore/min und Anstoß-Tore im
Training sind beim Drill nicht vergleichbar: Drill-Episoden enden nach 6 s.)

Teamspiel, 2v2-Duelle (500 Spiele à 300 s) und Team-Kennzahlen je Seite:

| A gegen B (2v2) | Tordifferenz/Spiel [95-%-KI] | Siege | Double-Commit A / B | Absicherung A / B | Mitspielerabstand A / B |
|---|---|---|---|---|---|
| sp_mode_2v2 gegen Start | **+8,02** [+7,71; +8,33] | 496 : 0 | 4,2 % / 12,5 % | 86 % / 56 % | 2205 / 1289 uu |
| sp_mode_2v2_tau05 gegen Start | **+11,31** [+10,92; +11,69] | 500 : 0 | 2,9 % / 11,7 % | 89 % / 56 % | 2486 / 1276 uu |
| sp_mode_2v2_tau05 gegen sp_mode_2v2 | +0,58 [+0,39; +0,77] | 245 : 156 | 5,3 % / 6,6 % | 82 % / 78 % | 2170 / 1888 uu |

Im 1v1 ändert τ 0,5 gegenüber τ 0,1 nichts: −0,03 [−0,16; +0,10].

**SPS, sauber gemessen** (PC ohne andere Last, je 3 × 15 Mio. Steps abwechselnd ab 6.037.692.544,
`results\phase_c_2026-09-28\bench_sps\`): 1v1 72.050 ± 260, mit 2v2-Anteil [3,1,0] 73.320 ± 290
Steps/s (+1,8 %). 2v2-Envs liefern pro Physik-Schritt doppelt so viele Samples; Env-Step- und
Inferenzzeit pro Iteration sind gleich.

**Entscheidungen:**

* **Anstoß-Drill (K1): behalten.** Jede Anstoß-Kennzahl ist besser als bei beiden Referenzläufen:
  gegen den Start 23 % erste Berührung (0 % / 4 %) und 30 : 13 Anstoß-Tore (7 : 509 / 17 : 128),
  Anstoß-Zeit im Training 3,00 s (3,45 / 3,10), gegen das Referenz-Ende 94 % zuerst (Wiederholung 79 %).
  Die Drift der Referenz in einen langsamen Anstoß tritt mit Drill nicht auf. Die Spielstärke liegt
  im Rauschen (+0,28 gegen den Start), in der gemeinsamen Ladder ist der Drill vorn. Grenzen: Der Bot
  kommt weiter 0,4–0,9 s nach einem Speedflip an; gegen Menschen hilft sofort nur das Skript (§8.4).
* **Potenzialbasiertes Shaping (E1): verwerfen.** Die Policy bricht zusammen: Nach 2 Mio. Steps
  steigt die Entropie von 2,80 auf 3,10, nach 40 Mio. auf 3,90, Clip-Fraction 0,03 → 0,001,
  Ballkontakt 0,028 → 0,001, 77 % der Episoden enden über NoTouch. Gleiches Muster wie K3 in Stufe 3,
  nur stärker: Die Potenzialdifferenz hat im Mittel ~0, die Returns verlieren fast alle Varianz,
  der Trainer teilt durch die übernommene Return-std und normiert Advantages nicht, der
  Entropie-Bonus gewinnt. Die Kalibrierung auf gleichen Betrag je Schritt (Faktor 8) reicht nicht.
  Ein neuer Versuch wäre nur mit Advantage-Normierung oder neu geschätzter Return-std sinnvoll.
* **Luftberührung (A1): verwerfen als Luftspiel-Hebel.** Das Ziel wird verfehlt: Aerials
  0,010 pro Spielerminute (Referenzen 0,013 / 0,012). Der Reward zahlt vor allem Sprungkontakte am
  Boden. Als Nebeneffekt bessere Konversion im Angriffsdrittel (0,237 gegen 0,136 / 0,201) und der
  stärkste Lauf gegen den Start (+0,74), beides aber im Trainingsrauschen.
* **2v2-Anteil (T1): behalten, wenn der Bot Teamspiel lernen soll.** 2v2 gegen den reinen
  1v1-Checkpoint +8 Tore/Spiel, 496 : 0 Siege, Double-Commit 12,5 → 4,2 %, Absicherung 56 → 86 %.
  1v1 ohne messbaren Verlust (+0,39 gegen den Start, im Bereich der Referenzläufe), SPS +1,8 %.
* **team_spirit 0,5 (T2): behalten.** Im 2v2 +0,58 [+0,39; +0,77] gegen τ 0,1, weniger
  Double-Commits (2,9 gegen 4,2 %) und weiteres Auseinanderstehen, im 1v1 neutral. Vorbehalt: Es
  gibt keine Wiederholung eines 2v2-Laufs, das Rauschen im 2v2 ist unbekannt.
* **Ecken-Schleife: offen.** Der einzige gezielte Versuch (E1) ist gescheitert; die Kennzahl
  „Drittel lang“ bewegt sich in keinem Lauf deutlich (0,39–0,43). Nächster Versuch: Torwert ×3
  (`sp_goal_x3`, vorbereitet, nicht gelaufen).

### 8.10 Hauptlauf-Config (Vorschlag, nicht gestartet)

Zwei Varianten, je nach Entscheidung in §8.11, beide checkpoint-kompatibel (Obs 257, 90 Aktionen,
512×3), festgelegt durch `test_main_run_proposals_add_only_the_kept_changes`:

* `train/configs/lucy_1v1_zero_sum_drill.json`: Hauptlauf-Config plus Anstoß-Drill (Gewicht 4).
  Lädt wie bisher aus `runs\lucy_1v1\checkpoints`.
* `train/configs/lucy_team_zero_sum.json`: dazu `mode_mix` [3,1,0] und `team_spirit` 0,5, eigener
  Ordner `runs\lucy_team\checkpoints` (vorher den neuesten Checkpoint aus `runs\lucy_1v1` dorthin
  kopieren; `runs\lucy_1v1` bleibt die reine 1v1-Linie und ein 1v1-Vergleichspartner). Drill und
  Teamspiel sind nur einzeln getestet, nicht zusammen.

Für Spiele gegen Menschen unabhängig davon: `$env:RLBOT_SCRIPTED_KICKOFF = "speedflip"` (§8.4).

### 8.11 Entscheidungsvorlage: 1v1-only oder Teamspiel dazulernen

| | 1v1-only (`lucy_1v1_zero_sum_drill`) | Teamspiel (`lucy_team_zero_sum`) |
|---|---|---|
| 2v2 im Spiel | untrainiert: Mitspieler-Gewichte auf Initialisierung, Double-Commit 12,5 %, Absicherung 56 % | nach 300 Mio. Steps +8 bis +11 Tore/Spiel gegen den 1v1-Checkpoint, Double-Commit 3–4 %, Absicherung 86–89 % |
| 1v1-Stärke | volle 1v1-Daten | nach 300 Mio. kein messbarer Verlust (+0,39 gegen den Start, Referenzläufe −3,24 / +0,19); langfristig nur 60 % der Samples 1v1, der 1v1-Fortschritt pro Stunde sinkt entsprechend, soweit das 2v2 nicht mitlernt |
| Durchsatz | 72.050 SPS | 73.320 SPS (+1,8 %) |
| Trainingszeit | – | für den gleichen 1v1-Anteil ~1,7× so lang (1 / 0,6) |
| Aufwand | Config wechseln | Config wechseln, Checkpoint kopieren, später τ auf 1,0 und ggf. 3v3 in den Mix |
| 3v3 | untrainiert | weiter untrainiert (Mix ohne 3v3) |

Empfehlung: **Teamspiel dazulernen**, wenn der Bot in 2v2 eingesetzt werden soll (so im Spieltest).
Der Gewinn im 2v2 ist riesig und sofort, die Kosten im 1v1 sind nicht messbar, der Durchsatz
unverändert. Das 1v1 bleibt über `runs\lucy_1v1` als Linie erhalten; mit den Duell-Werkzeugen lässt
sich regelmäßig prüfen, ob die Teamspiel-Linie im 1v1 zurückfällt. Nach 300–500 Mio. Steps mit
stabilen Team-Kennzahlen τ auf 1,0 anheben (§8.7).

### 8.12 Offen

* Ecken-Schleife: `sp_goal_x3` laufen lassen; die potenzialbasierte Form nur mit Advantage-Normierung
  neu versuchen.
* Luftspiel: `sp_aerial_share` und `sp_no_in_air` (vorbereitet). Der Bot hat im Selbstspiel kaum
  Aerial-Lagen; mehr Aerial-Starts sind der naheliegende nächste Hebel.
* `sp_kickoff_first_touch` (vorbereitet, nicht gelaufen).
* Trainingsrauschen: Mit zwei Referenzläufen ist es nur grob bekannt. Für knappe Entscheidungen
  wären drei oder mehr Wiederholungen nötig, oder das Duell gegen mehrere feste Gegner (Panel).
* Der geskriptete Anstoß ist im echten Rocket League noch ungetestet.

## 9. Geschwindigkeit (29./30.09.2026, lokal)

### 9.1 Auftrag, Werkzeug, Messmethode

Ziel: Training möglichst ~2× schneller, ohne dass die Lernqualität pro Step leidet. Obs 257,
90 Aktionen und Netze 512×3 bleiben (Checkpoint-kompatibel). Branch `claude/speed`, ein Commit je
Befund G1–G9 (Tabelle in `AUDIT_PROGRESS.md`; G9 ist ein Nebenbefund in `compare.py`). Der Hauptlauf lief bei Beginn nicht (letzte
Iteration 28.09. 16:34); `runs\lucy_1v1` wurde nur gelesen.

Die Upstream-Änderungen liegen im neuen gestapelten Patch `third_party/patches/rlgympppo_cpp_speed.patch`
(über dem Truncation-Patch; `tools/apply_patches.ps1` bestimmt den Stand von hinten,
`tools/export_upstream_patch.py` schreibt den Patch aus dem Klon). Jede Optimierung, die das Lernen
verändern kann, ist ein Config-Schalter mit Default = bisheriges Verhalten (wie R6).

Messwerkzeug `tools/bench_speed.py`: Jede Variante startet von einer Kopie von 6.037.692.544,
trainiert 15 Mio. Steps mit `lucy_1v1_zero_sum_drill.json` plus den Schaltern der Variante, die ersten
20 Iterationen fallen weg, drei Wiederholungen laufen **abwechselnd** (A B C A B C …). SPS = Steps
je Iteration ÷ `Total Iteration Time` (Wanduhr von Iteration zu Iteration, gilt auch mit
Überlappen). Rohdaten `results\speed_{g2,s1,s2,s3b,s4,s5}\` (`summary.md`, `runs.csv`, `metrics.csv` je
Lauf), Läufe `runs\speed_*`. Die Streuung zwischen Wiederholungen lag bei 0,2–2 %, in einer Sitzung
direkt nach einem Spiel bei ~8 % (s3b; die paarweisen Vergleiche blieben eindeutig).

**Absolute Zahlen schwanken zwischen Sitzungen, Vergleiche gelten nur innerhalb einer Sitzung.**
Am Morgen des 29.09. lief zero_sum mit dem alten Binary bei 72.050 SPS, am Mittag die Drill-Config
mit demselben Binary bei 62.900; der Drill selbst kostet nur 1,6 % (s1). Sammeln und Lernen waren
gleichmäßig ~0,1 s langsamer, also ein Maschinenzustand, kein Code-Effekt. Eine Messreihe (s3)
lief neben VALORANT und ist verworfen (`results\speed_s3\`, nicht ausgewertet).

### 9.2 Die Lücke zu Phase 0 (117.247 gegen ~63.000–73.000 SPS)

Phase 0 maß die **Benchmark-Config** (`bench/cpp/main.cpp`): Netz 3×256, DefaultObs (172), einfacher
Reward, 1 Epoche (3 Gradientenschritte je Iteration), kein Skill-Tracker, 16×64, ohne
`collection_during_learn`: 0,72 s Sammeln + 0,16 s Lernen je 100.000 Steps. Die Trainings-Config mit
dem alten Binary (G1-Profil, `runs\speed_g1probe_g1_1`): 0,92 s Sammeln + 0,67 s Lernen.

| Posten | Phase 0 | Training, alt | Ursache |
|---|---|---|---|
| PPO-Lernen | 0,16 s | 0,57 s | Netz 512×3 statt 256×3 (~4× FLOPs je Sample) und 6 statt 3 Gradientenschritte (`ppo_epochs` 2 × Puffer 3, H5). Darin 0,10 s Shuffle-Gather auf der CPU (ein Thread, 300.000 × 257 Floats je Epoche) und ~0,1 s Host→GPU-Kopien der Minibatches |
| Experience einfügen | (im Lernen) | 0,10 s | davon 0,07 s Umkopieren des vollen CPU-Puffers (`SubmitExperience`) |
| Sammeln | 0,72 s | 0,92 s | schwereres Env (Lucy-Rewards, 257er-Obs, Metrik-Callback 0,03 s), größeres Netz in der Inferenz, Skill-Eval und Iterations-Callback im Iterationsrest (~0,04 s) |

Zu den einzelnen Verdachtsmomenten:

* **Overlap:** kein Teil der Lücke. Auch die 117.247 sind ohne `collection_during_learn` gemessen;
  mit der Option waren es 115.661. Auf der GPU sperrt Upstream die Sammel-Threads während PPO
  ohnehin (§9.4).
* **Rewards/Obs:** stecken in `Env Step Time` (0,25–0,35 s je Thread und Iteration, samt Physik).
  Sie lassen sich nicht abschalten, ohne das Lernproblem zu ändern.
* **Metriken (PlayStats):** 0,008–0,013 s von 0,6–0,9 s Sammelzeit je Thread, der ganze
  Step-Callback 0,025–0,039 s. Das ist unter 1 % der Iteration, **seltener berechnen lohnt nicht.**
* **Anstoß-Drill:** 1,6 % (s1: 78.486 ohne gegen 77.224 SPS mit Drill).
* **Netz und Epochen:** der größte Posten. Beide lassen sich nicht ändern, ohne die Checkpoints zu
  brechen bzw. die Lern-Hyperparameter zu ändern (Phase 0 §7: `ppo_epochs` 1 +19 %, H5 offen).

### 9.3 Wo die Zeit hingeht (Profil G1, altes Verhalten)

G1 schreibt die Aufteilung in `metrics.csv`: Sammeln je Thread (`Infer Call`, `Traj Append`,
`Obs Tensor`, `Obs To Device`, `Env Step` mit `Step Callback` und `Play Stats`, `Agent Wait`),
Lernen (`Add Experience` mit `Exp Value Pred/GAE/Submit`, `PPO Shuffle/Minibatch/Optim/Param Copy`,
`Empty Cache`), Iterationsrest (`Prev Tail/Skill Eval/Iteration Callback/Save Time`).
`RLBOT_PROFILE_SYNC=1` wartet an den Phasengrenzen auf die GPU, damit GPU-Zeit der richtigen Phase
zugeordnet wird (nur zum Profilen; die Aufteilung änderte sich damit kaum).

| Teil | s je Iteration | Befund |
|---|---|---|
| Trajektorien anhängen | 0,36 (je Thread) | ~20 winzige Tensor-Operationen je Spieler und Schritt (`torch::tensor`, `index_copy_`, `unsqueeze`); der größte Einzelposten, reine CPU-Verwaltung. „Policy Infer Time“ enthielt ihn immer mit |
| GPU-Inferenz | 0,13 (je Thread) | Batch 128 Zeilen, die GPU-Arbeit je Aufruf ist klein; alle 16 Threads teilen sich den Standard-Stream, jedes `.cpu()` wartet |
| Env-Schritt | 0,35 (je Thread) | Physik, Obs, Rewards, Callback |
| Obs-Tensor / Kopie zur GPU | 0,04 / 0,01 | `FLIST2_TO_TENSOR` je Spiel plus `concat` / pageable H2D (Transfers sind kein Engpass) |
| Trajektorien zusammenfügen | 0,09 | `MultiAppend` über 2.048 Teilstücke |
| Shuffle | 0,10 | CPU-Gather in einem Thread |
| Minibatches | 0,46 | 12 × 50.000 Samples; H2D je Minibatch, fünf `.item()`-Synchronisationen je Minibatch |
| Puffer nachschieben | 0,07 | 2 × 205 MB Klonen/Kopieren auf der CPU |
| `emptyCache` | 0,004 | vernachlässigbar, nicht geändert |

Nach G2 verschob sich das Bild: Ohne das Anhängen stauen sich die Threads an der Inferenz
(`Infer Call` 0,13 → 0,30 s je Thread), die CPU ist dabei nur zu ~30 % ausgelastet. Das Sammeln hängt
seitdem an der serialisierten GPU-Inferenz, nicht an der Simulation. Mit Overlap (G5) wird die
Lernphase zum Engpass: PPO braucht unter GPU-Konkurrenz 0,55–0,74 statt 0,44 s.

### 9.4 Kandidaten und SPS

| # | Änderung | SPS vorher → nachher (Sitzung) | Einfluss aufs Lernen | Empfehlung |
|---|---|---|---|---|
| G2 | Trajektorien als Arrays statt Tensor-Ops | 62.896 → 78.015, **+24,0 %** (g2) | keiner: bitgleich, Test gegen den alten Pfad | übernehmen (immer an) |
| G3 | Warte-Schleifen prüfen `shouldRun` | – | keiner (Hänger beim Beenden, Voraussetzung für G5) | übernehmen (immer an) |
| G4 | `exp_buffer_on_device` | 77.224 → 90.652, **+17,4 %** (s1) | keiner: bitgleich (Test über 5 Iterationen inkl. vollem Puffer) | übernehmen |
| – | `collection_during_learn` wie bisher | 90.652 → 92.019, +1,5 % (s1, im Rauschen) | wenige Steps (~2 %) aus der Policy vor dem Update | allein wirkungslos |
| G5 | `infer_during_learn`, `collect_limit_factor` 1,0 | 89.948 → 112.233, **+24,8 %** (s2) | 89 % der Steps einer Iteration von der Policy vor dem letzten Update; KL 0,0031 → 0,0051, Clip-Fraction 3,0 → 5,2 % (gemessen gegen die sammelnde Policy) | Lernvergleich §9.6 |
| G5 | … mit Upstream-Limit 1,5 | → 139.780 (s1) | Iteration wächst auf ~148.700 Steps, weniger Updates je Sample | **nicht** vergleichbar, nicht verwenden |
| G6 | `tf32` (mit Overlap) | 112.233 → 128.205, +14,2 % (s2) | Numerik (10-Bit-Mantisse, auch in der Inferenz) | nur ohne AMP sinnvoll; zusätzlich zu AMP +1 % → nicht übernehmen |
| G7 | `autocast_learn` (BF16, ohne Grad-Scaler) | ohne Overlap 89.948 → 109.404, +21,6 %; mit Overlap 112.233 → 146.065, **+30,1 %** (s2) | Numerik der Vorwärtsrechnung: ohne Overlap KL 0,0031 → 0,0034, Clip 3,0 → 3,3 %; mit Overlap 0,0053 / 5,5 % | Lernvergleich §9.6 |
| G8 | `learner_high_priority_stream` | mit AMP 152.632 → 171.682, **+12,5 %** (s3b, jedes Paar +10 bis +15 %); ohne AMP +2,4 % (Rauschen) | Rechnung unverändert (nur Ablaufplanung); die Lernphase ist kürzer, also stammen weniger Steps aus der alten Policy (89 → 63 %, KL 0,0051 → 0,0042, Clip 5,3 → 4,3 %) | mit Overlap übernehmen |
| – | Threads × Spiele (Overlap+AMP+G8) | 16×64 167.978 → 16×96 186.028 (+10,7 %), 16×128 193.874 (+15,4 %), 16×160 197.558 (+17,6 %) (s4); 8×128 und 12×96 langsamer (s3b) | kürzere Trajektorienstücke je Iteration (16×64: ~49, 16×128: ~25, 16×160: ~20 Steps): die GAE (λ 0,95, γ 0,9954, Reichweite ~18 Steps) bootstrappt öfter vom Critic; sichtbar am Value Loss 0,214 → 0,197 / 0,178 / 0,164 | vorerst 16×64; mehr Spiele erst nach eigenem Lernvergleich |
| – | PlayStats seltener | – (Kosten < 1 %) | – | nicht nötig |

Nicht umgesetzt:

* **CUDA Graphs für die Inferenz:** Die Inferenz kostet nach G2/G5/G8 noch 0,18 s je Thread und
  Iteration (16×64), mit größeren Batches 0,08–0,12 s. Graphen bräuchten feste Batchgrößen je Thread
  (1v1/2v2-Mix) und eine graph-sichere Zufallszahlen-Behandlung für `multinomial`. Der Aufwand lohnt
  erst, wenn die Inferenz wieder der Engpass ist; derzeit ist es die Lernphase.
* **FP16/BF16-Inferenz:** Die Upstream-Option ist absichtlich deaktiviert („Potential cause of learning
  errors“). Die Inferenz ist latenz-, nicht rechenbegrenzt, und abweichende Log-Wahrscheinlichkeiten
  zwischen Inferenz und Lernen verzerren das PPO-Verhältnis. Kein Versuch.
* **Minibatch-Größe** (s5, Overlap+AMP+G8): 50.000 → 100.000 (ganzer Batch; die Minibatches werden
  vor dem Optimierer-Schritt ohnehin aufsummiert, die Rechnung bleibt bis auf Rundung gleich) ergibt
  181.479 → 160.524 SPS, **−11,5 %**. PPO wird schneller (0,33 → 0,26 s), aber die größeren Kernel
  verdrängen die Inferenz der Sammel-Threads (0,18 → 0,24 s je Thread). 50.000 bleibt; VRAM wäre mit
  4,9 von 12,2 GB (inkl. Desktop) kein Hindernis.

### 9.5 Lernvergleich (30.09.2026, je 300 Mio. Steps)

Frage: Leidet die Lernqualität pro Step unter Overlap (Daten aus der Policy vor dem letzten Update)
und BF16-Autocast? Aufbau wie Phase C: Start 6.037.692.544, Seed 123, 300 Mio. Steps,
`run_experiment.ps1` (Build `build\cpp_cu128_speed`, Git `778ca91`), Duelle je 1000 Spiele à 300 s,
Anstöße je 1000, gemeinsame Ladder (compare.py, 100 Spiele je Paarung). Referenz ist der Phase-C-Lauf
`sp_kickoff_drill` (= Hauptlauf-Config `lucy_1v1_zero_sum_drill` als Experiment). Weil zwei Läufe
derselben Config in Phase C bis 3,4 Tore/Spiel auseinanderlagen, lief dieselbe Config noch einmal mit
dem neuen Build und ohne Speed-Schalter (G2 ist bitgleich); ihr Abstand zur Referenz ist das Rauschen.
Treiber `results\speed_quality_2026-09-29\run_quality.ps1`, Vergleich `compare.md` dort,
Einzelergebnisse `results\exp_{speed_overlap_amp,speed_overlap,replicate_sp_kickoff_drill}_2026-09-30_*`.

| Lauf | Wanduhr 300 Mio. | SPS (letztes Fünftel) | Duell gg. Start | Duell gg. Referenz-Ende | Ladder μ−3σ | Anstoß zuerst gg. Start |
|---|---|---|---|---|---|---|
| Referenz `sp_kickoff_drill` (Phase C, altes Binary) | 4.529 s (teils neben VALORANT) | 70.685 | +0,28 [+0,18; +0,39] | – | 22,44 | 22,6 % |
| Wiederholung, gleiche Config, neuer Build | 3.465 s | 88.315 | +0,67 [+0,56; +0,77] | +0,29 [+0,19; +0,39] | 23,86 | 74,8 % |
| `speed_overlap` (G4+G5+G8, FP32) | 2.291 s | 138.823 | +0,58 [+0,47; +0,68] | +0,38 [+0,27; +0,48] | 24,17 | 53,5 % |
| `speed_overlap_amp` (+G7) | **1.841 s** | **167.822** | +0,81 [+0,70; +0,92] | +0,58 [+0,46; +0,69] | **24,35** | 52,9 % |

Trainingsmetriken im letzten Fünftel (Referenz / Wiederholung / Overlap / Overlap+AMP): Entropie
2,81 / 2,79 / 2,79 / 2,77; Tor-Anteil der Episodenenden 0,645 / 0,640 / 0,639 / 0,636; Value Loss
0,192 / 0,190 / 0,198 / 0,203; Anstoß-Zeit bis zur ersten Berührung 3,00 / 3,00 / 3,12 / 3,06 s;
Tore/min 0,57 / 0,63 / 0,69 / 0,69; Drittel-Konversion 0,124 / 0,135 / 0,145 / 0,148. KL und
Clip-Fraction sind mit Overlap höher (0,0031 → 0,0038 / 0,0042; 3,0 → 3,9 / 4,2 %), weil sie gegen die
sammelnde Policy gemessen werden, die eine Version älter ist; die Entropie fällt dabei nicht.

Bewertung (vorsichtig, eine Wiederholung je Variante):

* **Kein Hinweis auf Schaden.** Beide Speed-Läufe schlagen das Referenz-Ende und den Start, liegen in
  der gemeinsamen Ladder vor beiden Läufen mit unveränderter Config und in jeder Trainingskennzahl im
  Bereich der beiden Referenzläufe. compare.py nennt beide Effekte „im Trainingsrauschen“ (Abstand der
  Wiederholung zur Referenz 0,29–0,39 Tore/Spiel): also gleich gut, nicht nachweislich besser.
* **Anstoß:** Die beiden Läufe mit derselben Config liegen bei 22,6 % und 74,8 % „zuerst am Ball“
  gegen den Start; die Speed-Läufe mit 53 % dazwischen. Die Anstoß-Zeit im Training ist 0,06–0,12 s
  länger, bei einer Streuung zwischen Läufen von ~0,3 s (Phase C). Beobachten, kein Befund.
* **Overlap und AMP einzeln:** Overlap allein (FP32) und mit AMP zeigen dasselbe Bild; AMP verschlechtert
  nichts messbar.

### 9.6 Vorschlag Hauptlauf (nicht gestartet)

`train/configs/lucy_1v1_zero_sum_drill_fast.json` = `lucy_1v1_zero_sum_drill.json` plus sechs
Learner-Schalter: `exp_buffer_on_device`, `collection_during_learn`, `infer_during_learn`,
`collect_limit_factor` 1,0, `autocast_learn`, `learner_high_priority_stream`; 16×64 wie bisher, gleicher
Checkpoint-Ordner (Test `test_fast_main_run_proposal_adds_only_speed_switches`). Gemessen ~168.000 SPS im
15-Mio.-Fenster und 167.822 im 300-Mio.-Lauf, gegen ~88.000 mit dem neuen Build ohne Schalter und
~63.000–73.000 mit dem alten Binary: **Faktor ~2,3–2,7 gegenüber dem bisherigen Hauptlauf, 1,9 gegenüber
dem neuen Build allein.** Voraussetzung: `build\cpp_cu128` aus `claude/speed` neu bauen
(`run_all_checks.ps1` tut das). Mit der alten Config verhält sich der neue Build wie bisher (G2
bitgleich, alle Schalter Default aus), ist aber durch G2 schon ~25 % schneller.

Rückweg ohne Risiko für das Lernen: nur `exp_buffer_on_device` (bitgleich) setzen, ~90.000 SPS.

Mehr Spiele je Thread (16×96 bis 16×160, weitere +11 bis +18 %) sind nicht im Vorschlag: Sie verkürzen
die Trajektorienstücke je Iteration und ändern damit die GAE-Ziele (Value Loss 0,214 → 0,164). Erst mit
eigenem Lernvergleich.

### 9.7 Offen

* Mehr Spiele je Thread (s. o.), eigener 300-Mio.-Vergleich.
* Längerer Beleg: Der Vergleich deckt 300 Mio. Steps ab; im Hauptlauf die Duell-Werkzeuge regelmäßig
  gegen ältere Checkpoints laufen lassen (wie in §8.11 vorgesehen).
* CUDA Graphs für die Inferenz, falls sie nach größeren Batches wieder zum Engpass wird.
* Der Phase-0-Benchmark (`bench_cpp_sps`) wurde nicht neu gemessen; die Lücke ist über die Bauteile
  erklärt (§9.2), nicht über eine neue Referenzmessung.

## 10. Hauptlauf-Betrieb (30.09.2026, lokal)

### 10.1 Auftrag und Ausgangslage

Der Hauptlauf sollte mit `lucy_1v1_zero_sum_drill_fast.json` laufen (~180.000–200.000 SPS). Auftrag: drei
Verbesserungen (Checkpoint-Historie/Skill-Tracker, Regressions-Check, Autocast-Warnungen), danach den
Hauptlauf selbst neu starten und mindestens eine Stunde beobachten; keine lernrelevanten Werte ändern.
Branch `claude/hauptlauf-betrieb` (von `main` @ `3308fed`), Commits B1–B4.

Befund beim Start der Arbeit (12:45): **Der Hauptlauf lief nicht.** Er war zweimal kurz gestartet worden
(zuletzt 12:43:09, `config_used.json`, `_git` 2ebc0c7) und endete zuletzt um 12:44:25 bei 6.046.918.784
Steps, jeweils ohne neuen Checkpoint (der neueste blieb 6.037.692.544) und ohne Absturzeintrag im
Ereignisprotokoll. Die ~9 Mio. Steps fehlen; `metrics.csv` enthält die 88 und 92 Zeilen der beiden
Kurzläufe. Punkt „laufenden Hauptlauf sauber stoppen“ entfiel damit.

### 10.2 B1: Checkpoint-Historie und Skill-Tracker

Bei ~180.000 SPS deckten 50 Checkpoints à 25 Mio. Steps nur 1,25 Mrd. Steps (~2 h) ab. Der Skill-Tracker
(20 Versionen im Abstand von 500 Mio.) lädt alte Versionen beim Start aus dem Checkpoint-Ordner und fand
deshalb 3 von 20. Checkpoint gemessen: 15,6 MB (Policy 2,8, Critic 2,6, Optimizer 5,6 + 5,3 MB).

| Wert | vorher | jetzt | Folge |
|---|---|---|---|
| `timesteps_per_save` | 25 Mio. | **50 Mio.** | alle ~4,6 min; ein Absturz kostet höchstens so viel, ein geplanter Stopp mit `--save-on-exit` nichts |
| `checkpoints_to_keep` | 50 | **200** | Historie 10 Mrd. Steps (~15 h bei 180.000 SPS), 3,1 GB (Reserve bleibt weit über 10 GB) |
| `skill_timesteps_per_version` | 500 Mio. | **250 Mio.** | 20 Versionen über 5 Mrd. Steps, alle innerhalb der Historie und auf gespeicherten Checkpoints; beim Neustart jetzt 5 statt 3 gefunden, voll nach ~7,7 h |

Mehr Gegner kosten keine SPS: Die Skill-Eval spielt eine feste Spielzeit (`simTime` verteilt auf
`numEnvs` Spiele), die Zahl der Versionen bestimmt nur, gegen wen. Gemessen (`results\speed_b1skill`, je
3 × 15 Mio. Steps mit 50 kopierten Checkpoints): 3 geladene Versionen 179.236 SPS, 20 Versionen 179.626
(+0,2 %), Skill-Eval 0,020 s je Iteration in beiden.

### 10.3 B2: Regressions-Check

`tools/regression_check.py`: neuester vollständiger Checkpoint gegen den vollständigen, der am nächsten
an „neuester − 1 Mrd.“ liegt. Beide Policies werden zuerst kopiert (Lauf nur gelesen, Rotation kann nichts
wegnehmen). Duell 1000 × 300 s mit `--threads 2`, Anstöße mit `eval/kickoff_eval.py` (500, ein
Torch-Thread). Urteil nach dem 95-%-Intervall der Tordifferenz: besser / gleich / schlechter; Verlauf mit
Datum in `results\regression\lucy_1v1_history.md` und `.csv`.

### 10.4 B3: Autocast-Warnungen

`RG_AUTOCAST_ON/OFF` riefen die veralteten Wrapper `at::autocast::set_enabled`,
`set_autocast_gpu_dtype`, `set_autocast_cpu_dtype`; jeder löst zur Laufzeit `TORCH_WARN_DEPRECATION` aus
(4 je Minibatch, ~48 Zeilen je Iteration). Jetzt `set_autocast_enabled(at::kCUDA, …)` und
`set_autocast_dtype(at::kCUDA/at::kCPU, …)`, genau die Aufrufe, an die die Wrapper weiterleiten. Test:
eigener c10-Warning-Handler, Zustand und ein BF16-Produkt bitgleich zur alten API, echter PPO-Schritt
ohne Warnung; im Hauptlauf-Log nach dem Neustart 0 Deprecation-Zeilen.

### 10.5 B4: Starten, Stoppen, Prüfen

`tools\local\start_main_run.ps1` (Aufgabenplanung, eigenes Fenster, keine 72-h-Grenze; Eltern-Prozess
ist die Aufgabenplanung, nicht die startende Sitzung), `run_main.ps1` (Trainer mit `--stop-file
runs\hauptlauf\STOP --save-on-exit`, Log `runs\hauptlauf\train_<datum>.log`), `stop_main_run.ps1`
(Stop-Datei, wartet, nie hart), `main_run_status.py` (Grenzwerte über die letzten 200 Iterationen).
Befehle: LOCAL_RUNBOOK §5d.

Hinweis zu „ep_end_goal nahe 1“: Mit dem Anstoß-Drill enden ~36 % der Episoden planmäßig nach 6 s
(`ep_end_drill`), `ep_end_goal` liegt deshalb bei ~0,64. Geprüft wird der Tor-Anteil der übrigen Episoden,
`ep_end_goal / (1 − ep_end_drill)`; er liegt bei 1,000.

### 10.6 Neustart (30.09.2026, 13:15)

Prüfpaket vorher grün (`5b18265`, sauber): 6/6 Schritte in 5,9 min, C++ 119/119 und Python 207/207 je
zweimal, Golden-Fixtures unverändert, Smoke und Deployment-Smoke OK; `build\cpp_cu128` dabei neu gebaut
(der Hauptlauf lief nicht, die .exe war frei). Start mit `start_main_run.ps1` um 13:15:16: train_bot.exe
(PID 32880) mit `--stop-file runs\hauptlauf\STOP --save-on-exit`, Eltern-Prozess powershell.exe, dessen
Eltern svchost.exe (Aufgabenplanung). Geprüft: lädt 6.037.692.544, Skill-Tracker findet 5 Versionen
(250 Mio. Abstand), Lern-Stream hoher Priorität aktiv, nach 153 Iterationen 0 Deprecation-Zeilen im Log
(vorher dutzende je Iteration), 183.494 SPS. Log `runs\hauptlauf\train_2026-09-30_131516.log`
(~5,8 MB je Stunde).

### 10.7 Beobachtung (Mittel der letzten 200 Iterationen, `main_run_status.py`, `runs\hauptlauf\status.csv`)

| Zeit | Steps | SPS | Entropie | KL | Clip | Value Loss | ep_end_goal (ohne Drill) | Timeouts | Checkpoints | Hinweis |
|---|---|---|---|---|---|---|---|---|---|---|
| 13:17 | 6.058.854.656 | 177.428 | 2,770 | 0,0043 | 0,043 | 0,216 | 0,630 (1,000) | 0,0003 | 50 | Fenster noch mit Zeilen der Kurzläufe |
| 13:31 | 6.203.574.016 | 178.763 | 2,768 | 0,0043 | 0,044 | 0,206 | 0,642 (1,000) | 0,0001 | 53 | |
| 13:46 | 6.364.951.296 | 178.343 | 2,775 | 0,0043 | 0,043 | 0,205 | 0,639 (1,000) | 0,0000 | 56 | |
| 14:01 | 6.528.225.408 | 179.860 | 2,797 | 0,0043 | 0,043 | 0,206 | 0,630 (1,000) | 0,0002 | 59 | |
| 14:16 | 6.689.199.360 | 179.015 | 2,844 | 0,0042 | 0,042 | 0,189 | 0,635 (0,999) | 0,0006 | 63 | |
| 14:31 | 6.851.175.680 | 179.091 | 2,842 | 0,0042 | 0,042 | 0,190 | 0,633 (1,000) | 0,0002 | 66 | |
| 14:46 | 7.012.145.792 | 179.542 | 2,848 | 0,0042 | 0,042 | 0,191 | 0,639 (1,000) | 0,0002 | 69 | |
| 15:01 | 7.160.859.648 | 159.389 | 2,861 | 0,0044 | 0,043 | 0,187 | 0,642 (1,000) | 0,0001 | 72 | Regressions-Check läuft (2 Threads) |
| 15:16 | 7.307.261.312 | 169.390 | 2,870 | 0,0042 | 0,042 | 0,177 | 0,641 (1,000) | 0,0002 | 75 | Check endete 15:16 |

Alle Grenzwerte in jeder Prüfung eingehalten, kein Spiel nebenher. Die Entropie steigt langsam
(2,77 → 2,87 in 1,3 Mrd. Steps; im Phase-C-Drill-Lauf 2,77 → 2,81 in 300 Mio.), weit über der
Grenze 2,5 — beobachten. KL und Clip liegen wegen des Overlaps über dem alten Lauf (0,0031 / 3,0 %,
§9.5) und sind konstant. Die Anstoß-Zeit bis zur ersten Berührung fällt im Training von 3,6 auf 2,65 s.

### 10.8 Regressions-Check nach 1 Mrd. Steps

`tools/regression_check.py` um 14:48 (1000 Spiele, 2 Threads, 500 Anstöße, 27 min):

| neu | alt | Urteil | Tordifferenz/Spiel [95-%-KI] | Siege neu:alt | Anstoß zuerst (neu) | erste Berührung neu/alt |
|---|---|---|---|---|---|---|
| 7.038.624.000 | 6.037.692.544 | **besser** | **+1,69** [+1,58; +1,80] | 757:96 (147 remis) | 94,8 % [92,5; 96,4] | 2,56 s / 3,17 s |

Der neue Stand ist nach 1 Mrd. Steps mit der Fast-Config deutlich stärker, vor allem beim Anstoß (der Drill
wirkt). Kosten des Checks für das Training: 179.500 → 159.400 SPS (−11 %) für ~27 min; mit `--threads 1`
wäre es etwa die Hälfte bei doppelter Dauer.

### 10.9 Offen

* Entropie-Anstieg weiter beobachten (Grenze 2,5 ist weit weg; ein Anstieg über ~3,0 wäre ein Hinweis
  wie bei K3/E1, dass der Entropie-Bonus relativ zu stark wird).
* Das Trainer-Log wächst ~140 MB am Tag; alte Logs in `runs\hauptlauf\` gelegentlich löschen.
* `metrics.csv` enthält die zwei Kurzläufe vom Mittag (88 und 92 Iterationen ab 6,0378 Mrd.) vor dem
  Neustart; Auswertungen über Steps sehen den Bereich 6,038–6,047 Mrd. dreifach.

### 10.10 Weiterer Verlauf bis 10 Mrd. Steps (30.09./01.10.2026)

* Der Lauf von 13:15 wurde um 15:28 durch Schließen des Fensters beendet, ein neuer lief ab 15:31 (ab
  7.438.952.192) und endete um 20:19 ebenfalls durch Schließen des Fensters bei 9.737.629.440 Steps
  („forrtl: error (200): program aborting due to window-CLOSE event“). Das Schließen beendet den Trainer
  sofort, ohne End-Checkpoint: verloren waren ~8 und ~47 Mio. Steps (höchstens ein Speicherabstand,
  50 Mio.). Sauber und ohne Verlust stoppt nur die Stop-Datei (`tools\local\stop_main_run.ps1`).
* Verlauf 6,0–9,74 Mrd. (Mittel je 250 Mio.): Entropie 2,77 → 2,97, ab 8,5 Mrd. flach bei ~2,97; KL
  0,0041–0,0043 und Clip 3,9–4,3 % konstant; Value Loss 0,211 → 0,158; Tor-Anteil ohne Drill 1,000;
  Anstoß-Zeit im Training 3,2 → 2,6 s (bis 7,0 Mrd.), danach wieder ~3,0–3,2 s (Spitze 4,1 s bei
  8,5–8,75 Mrd.). Ab ~8,25 Mrd. lag die SPS bei 107.000–123.000 (vermutlich Spiele nebenher).
* Neustart am 30.09. um 23:42 mit `start_main_run.ps1` ab 9.690.605.696: der Skill-Tracker findet jetzt
  alle 20 Versionen, ~200.000 SPS, keine Warnungen. Beobachtung bis 00:42 (Tabelle unten): alle
  Grenzwerte eingehalten.
* Zweiter Regressions-Check (23:44): 9.690.605.696 gegen 8.690.024.064 **besser**, +1,48 [+1,33; +1,64]
  Tore/Spiel, Siege 655:211 (134 remis), Anstoß zuerst 71,8 % [67,7; 75,6], erste Berührung 2,82 s gegen
  2,80 s. Der Lauf wird also trotz steigender Entropie und wieder längerer Anstoß-Zeit im Selbstspiel
  weiter deutlich stärker; die Anstoß-Überlegenheit gegen den 1 Mrd. älteren Stand ist kleiner als beim
  ersten Check (94,8 %).

| Zeit | Steps | SPS | Entropie | KL | Clip | Value Loss | Tor-Anteil ohne Drill | Hinweis |
|---|---|---|---|---|---|---|---|---|
| 09-30 23:58 | 9.842.517.376 | 172.330 | 2,975 | 0,0043 | 0,042 | 0,162 | 1,000 | Regressions-Check läuft |
| 10-01 00:11 | 9.981.309.440 | 172.041 | 2,979 | 0,0042 | 0,041 | 0,159 | 1,000 | Fenster enthält noch Iterationen während des Checks |
| 10-01 00:27 | 10.154.514.944 | 191.782 | 2,952 | 0,0042 | 0,040 | 0,158 | 1,000 |  |
| 10-01 00:42 | 10.327.127.168 | 190.321 | 2,967 | 0,0042 | 0,040 | 0,156 | 1,000 |  |

Offen (zusätzlich zu §10.9): Die Entropie liegt knapp unter 3,0; steigt sie weiter, wäre das der Punkt, an
dem der Entropie-Bonus relativ zu stark wird (Muster K3/E1). Solange die Regressions-Checks „besser“
melden, ist das kein Handlungsbedarf. Optional: train_bot.exe könnte das Schließen des Fensters abfangen
(CTRL_CLOSE_EVENT, ~5 s Zeit) und noch speichern; nicht umgesetzt.

## 11. Spieltest 2 (01.10.2026, lokal)

### 11.1 Anlass und Urteil des Nutzers

Der Hauptlauf stand am 01.10. um 12:30 bei 16,97 Mrd. Steps (hart beendet, neuester Checkpoint
16.946.547.712). Der Nutzer hat diesen Stand in Rocket League gegen den Stand des ersten Spieltests
(6.037.692.544) spielen lassen (zweiter Bot-Eintrag `deploy/rlbot_alt/bot.toml`, B6). Urteil: Luftspiel
und Anstoß (ohne Skript) sind „nicht wirklich besser geworden“, der Rest schon.

### 11.2 Messungen dazu

**Gesamtstärke.** Duell 16.946.547.712 gegen 6.037.692.544 (1000 Spiele, beide ziehen wie im Training):
+8,43 [+8,28; +8,58] Tore/Spiel, 1000:0 Siege. Ein argmax-gegen-argmax-Duell hat nur 10 verschiedene
Spiele (5 Anstoßpositionen × 2 Seiten) und wurde nicht gewertet.

**Fortschritt je Milliarde Steps wird klein** (Regressions-Checks, `results/regression/lucy_1v1_history.md`):

| neu gegen alt | Tordifferenz/Spiel | Siege | Anstoß: neu zuerst | erste Berührung neu / alt |
|---|---|---|---|---|
| 7,04 gegen 6,04 Mrd. | +1,69 [+1,58; +1,80] | 757:96 | 94,8 % | 2,56 s / 3,17 s |
| 9,69 gegen 8,69 Mrd. | +1,48 [+1,33; +1,64] | 655:211 | 71,8 % | 2,82 s / 2,80 s |
| 16,95 gegen 15,95 Mrd. | +0,19 [+0,03; +0,34] | 441:406 | 61,4 % | 4,52 s / 3,26 s |
| 17,10 gegen 15,10 Mrd. | +0,69 [+0,53; +0,85] | 551:324 | 17,2 % | 3,47 s / 3,50 s |

**Anstoß im Selbstspiel** (Trainingsmetriken des Hauptlaufs, Mittel je Milliarde Steps):

| Mrd. Steps | 6–7 | 9–10 | 12–13 | 14–15 | 15–16 | 16–17 |
|---|---|---|---|---|---|---|
| erste Berührung (s) | 2,80 | 3,18 | 3,03 | 3,52 | 4,51 | 4,96 |
| Tempo bei der Berührung (uu/s) | 1445 | 1433 | 1473 | 1283 | 1091 | 1139 |
| Boost verbraucht | 37,6 | 29,5 | 32,4 | 25,0 | 20,3 | 17,7 |
| Anstöße ohne Berührung | 0,0 % | 0,0 % | 0,0 % | 0,1 % | 10,4 % | 28,6 % |

Der Umschlag liegt bei ~15,25 Mrd. Steps (Mittel je 250 Mio.: unberührt 0,1 % → 5,8 % → 20 %, zuletzt
26–36 %). Der Anstoß-Drill endet nach 6 s; „unberührt“ heißt, dass in dieser Zeit keiner am Ball war.

Was der Bot tut (`eval/kickoff_eval.py --trajectory`, argmax wie im Spiel, 16.946.547.712):

* gegen sich selbst: Boost 0–3, Spitze 1320–1410 uu/s, erste Berührung 3,0–3,6 s; beim Mittel-Anstoß
  berührt keiner den Ball;
* gegen 6.037.692.544 (fährt mit bis zu 45 Boost an): derselbe Bot fährt auf vier von fünf Positionen
  mit 40–47 Boost bis 2300 uu/s und ist in 2,4–3,0 s zuerst am Ball (500 Anstöße argmax: 100 % zuerst, das sind 10 verschiedene Abläufe;
  gezogen 55,4 % [51,0; 59,7], Tore in 10 s 28:10).

Der Bot kann also schnell anfahren, tut es aber nur, wenn der Gegner Druck macht. Ein Speedflip
(1,9–2,5 s, §8.4) bleibt schneller. Gegen den 2 Mrd. älteren Stand (15,10 Mrd., fährt noch normal an)
verliert der aktuelle Stand 83 % der ersten Berührungen.

**Wahrscheinliche Ursache (nicht per Experiment geprüft):** `save_boost` zahlt 0,3·√Boost je Step, mit
Zero-Sum also die Differenz zum Gegner. Wer beim Anstoß seine 33 Boost verbraucht und der Gegner
nicht, verliert bis zum nächsten Pad 0,17 je Step, das sind 2,6 je Sekunde; ein Tor ist 10 wert.
Alle anderen Shaping-Terme sind beim Anstoß symmetrisch und heben sich im Zero-Sum auf,
`velocity_player_to_ball` bringt dem Schnelleren höchstens ~0,04 je Step. Der Verlauf passt dazu:
Der Boost-Verbrauch beim Anstoß sinkt seit 6 Mrd. Steps fast stetig (37,6 → 17,7).

**Luftspiel.** Ballkontakte mit Ball über 450 uu je Spielerminute: im Duell 0,40 (16,95 Mrd.) gegen 0,012
(6,04 Mrd.), im Training 0,02 (6–7 Mrd.) → 0,34 (9–10 Mrd.) → 0,69 (16–17 Mrd.); mittlere Ballhöhe bei
Luftkontakten 160 → 200 uu. Das ist ein Kontakt alle 1,5–2,5 Minuten und im Spiel kaum zu sehen.
Aerial-Startzustände haben mit dem Drill nur noch 0,5 von 11,5 Gewichtsanteilen (~4 % der Episoden).

### 11.3 Experimente (vom Nutzer freigegeben)

Je 300 Mio. Steps ab 17.097.466.368 (sauberer Stopp des Hauptlaufs 13:13), Seed 123, Build `d857d26`,
nacheinander als Aufgabe „RLbot Experimente“; Referenz ist die laufende Hauptlauf-Config als Experiment
(`sp2_reference.json`). Während des Referenzlaufs liefen die beiden letzten Regressions-Checks mit
(SPS 177.000 statt 190.000–198.000, gleiche Step-Zahl). `results/compare_sp2.md`.

| | Referenz | `kickoff_first_touch` 2,0 | `state_setters.aerial` 0,5 → 2,0 |
|---|---|---|---|
| Duell Ende gegen Referenz-Ende | – | **+0,67** [+0,51; +0,83] | **+0,53** [+0,37; +0,69] |
| Duell Ende gegen den eigenen Start | **−1,25** [−1,42; −1,08] | +0,03 [−0,13; +0,19] | +0,30 [+0,14; +0,46] |
| TrueSkill gemeinsame Ladder (mu; Start 25,67) | 25,14 | 25,45 | 25,93 |
| Training: Anstöße ohne Berührung | 16,6 % | 4,3 % (Fünftel 2–4: unter 1 %) | 8,6 % |
| Training: erste Berührung / Tempo / Boost | 4,41 s / 1214 / 23,0 | 4,06 s / 1302 / 23,4 | 4,45 s / 1156 / 19,3 |
| Anstoß gegen 15,10 Mrd. zuerst (500, gezogen; Start: 17,2 %) | 3,6 % | 16,4 % | 32,4 % |
| Anstoß argmax gegen sich selbst: unberührt (von 5), Boost | 4, 0 | 3, 0 | 0, 0 |
| Training: Aerials je Minute (Fünftel 1 → 5) | 0,72 → 0,72 | 0,77 → 0,75 | 0,80 → 0,78 |
| Duell gegen Referenz-Ende: Aerials je Minute | – | 0,65 : 0,70 | 0,68 : 0,72 |

Lesart:

* **Das Trainingsrauschen ist groß.** Der unveränderte Referenzlauf ist nach 300 Mio. Steps 1,25
  Tore/Spiel schwächer als sein eigener Start; sein Anstoß ist fast ganz passiv geworden (zuerst am Ball
  4 % gegen den Start, argmax 4 von 5 Positionen unberührt). Die Vorsprünge der beiden Experimente auf
  das Referenz-Ende kommen zum großen Teil daher. Einzelne Checkpoints des Hauptlaufs schwanken also um
  etwa ±1 Tor/Spiel, je nachdem, wo der Anstoß gerade steht; das erklärt auch die kleinen und
  uneinheitlichen Regressions-Checks seit 15 Mrd.
* **Anstoß-Reward:** nimmt das gemeinsame Abwarten weitgehend weg (unberührt 16,6 % → 4,3 %) und hält
  die Stärke (±0 gegen den Start, wo die Referenz 1,25 verliert). Schneller wird der Anstoß in 300 Mio.
  Steps nicht: weiter ohne Boost, argmax bleiben 3 von 5 Positionen unberührt. +2 für die erste
  Berührung ist klein gegen den `save_boost`-Nachteil des Boostens.
* **Mehr Aerial-Starts:** kein Lerneffekt sichtbar. Die höhere Rate im Training (0,78 gegen 0,72) ist von
  Beginn an da (mehr Aerial-Szenen in der Mischung) und steigt im Lauf nicht; im Duell hat das Experiment
  nicht mehr Aerials als die Referenz. Nicht schädlich (+0,30 gegen den Start).
* Beide Hebel schaden nicht, lösen das jeweilige Problem aber in 300 Mio. Steps auch nicht.

### 11.4 Stand und offene Entscheidung

Der Hauptlauf läuft seit 15:25 wieder mit unveränderter Config ab 17.097.466.368 (~182.000 SPS, alle
Grenzwerte eingehalten). An der Config wurde nichts geändert; die Entscheidung liegt beim Nutzer.
Vorschläge:

1. `kickoff_first_touch` 2,0 in die Hauptlauf-Config übernehmen: stabilisiert den Anstoß gegen das
   Abwarten, kein Nachteil gemessen.
2. Die `save_boost`-Vermutung prüfen (ein Lauf, ~45 min): Referenz plus `save_boost` 0,3 → 0,1, oder der
   Anstoß-Reward mit deutlich höherem Wert. Erwartung: Boost-Verbrauch und Tempo beim Anstoß steigen.
3. Luftspiel: `sp_air_touch` (Reward für Luftkontakte, skaliert mit der Ballhöhe) als nächstes Experiment
   statt mehr Aerial-Starts; dazu ein längerer Lauf, 300 Mio. Steps sind für Aerials zu kurz.
4. Wegen des Rauschens: Entscheidungen an den gezielten Kennzahlen festmachen (Boost/Tempo/Zeit beim
   Anstoß, Aerials je Minute) und beim Duell eine Wiederholung der Referenz mitlaufen lassen.

Betrieb: `start_main_run.ps1` scheiterte nach dem sauberen Stopp („nach 90 s nicht gestartet“), weil das
alte Fenster die Aufgabe im Zustand „Running“ hielt; behoben (B7). `tools/export_policy.py` lief als
Skript nicht (B5). Für Spiele gegen Menschen bleibt `RLBOT_SCRIPTED_KICKOFF=speedflip` der schnellste Weg.

### 11.5 Zweite Runde: `save_boost` als Ursache (01./02.10.2026, vom Nutzer freigegeben)

Zwei weitere Läufe ab demselben Start (17.097.466.368, 300 Mio. Steps, Seed 123, Build `8508dfa`) gegen die
Referenz aus §11.3: `sp2_save_boost_01` (`save_boost` 0,3 → 0,1) und `sp2_save_boost_01_kickoff` (dazu
`kickoff_first_touch` 2,0). Der erste Lauf wurde am 01.10. bei der Anstoß-Auswertung versehentlich
abgebrochen (Training und beide Duelle waren fertig, Anstoß-Auswertung und Zusammenfassung von Hand mit
denselben Aufrufen nachgeholt, ohne Lauf-Ladder), der zweite lief am 02.10. ab 07:07. Der Hauptlauf stand
dadurch von 01.10. 15:52 bis 02.10. 08:00. `results/compare_sp2b.md`.

| | Referenz | nur `kickoff_first_touch` 2,0 | nur `save_boost` 0,1 | beides |
|---|---|---|---|---|
| Duell Ende gegen Referenz-Ende | – | +0,67 [+0,51; +0,83] | −0,30 [−0,47; −0,13] | **+0,22** [+0,06; +0,38] |
| Duell Ende gegen den eigenen Start | −1,25 [−1,42; −1,08] | +0,03 [−0,13; +0,19] | −0,60 [−0,75; −0,44] | **+0,14** [−0,03; +0,30] |
| davon Tore in 10 s nach Anstoß (Ende : Start, 1000 Spiele) | 111 : 988 | 93 : 113 | 169 : 423 | **354 : 126** |
| TrueSkill gemeinsame Ladder (mu; Start 25,66) | 25,21 | 25,56 | 25,07 | 25,64 |
| Training: Anstöße ohne Berührung (letztes Fünftel) | 16,6 % | 4,3 % | 6,0 % | **0,0 %** (ab Fünftel 3) |
| Training: erste Berührung / Tempo / Boost (letztes Fünftel) | 4,41 s / 1214 / 23,0 | 4,06 s / 1302 / 23,4 | 4,03 s / 1238 / 30,3 | 3,89 s / 1182 / 24,0 |
| Anstoß gegen 15,10 Mrd. zuerst (500, gezogen; Start 17,2 %) | 3,6 % | 16,4 % | 24,8 % | **58,0 %** |
| Anstoß argmax gegen sich selbst: unberührt (von 5) | 4 | 3 | 4 | **0** |
| Boost-Vorrat im Spiel (`boost_held`) | 0,43 | 0,43 | 0,35 | 0,36 |
| Entropie (letztes Fünftel) | 2,97 | 2,99 | 3,08 | 3,04 |
| Aerials je Minute im Training | 0,72 | 0,75 | 0,68 | 0,69 |

Lesart:

* **Die Schwankung der Gesamtstärke ist der Anstoß.** Das Referenz-Ende kassiert gegen seinen Start in
  1000 Spielen 988 Tore binnen 10 s nach einem Anstoß und schießt 111; das allein sind −0,88 von −1,25
  Toren/Spiel.
* **`save_boost` ist ein Teil der Ursache, aber nicht allein.** Mit 0,1 steigt der Boost-Verbrauch beim
  Anstoß (25,7 → 30,3 im Lauf, Referenz ~23), das Abwarten geht zurück. Ohne den Anstoß-Reward bleibt der
  Anstoß im argmax-Selbstspiel aber passiv (4 von 5 unberührt), und der Lauf verliert Anstoß-Tore (169:423).
* **Beides zusammen wirkt am besten:** kein unberührter Anstoß mehr, im argmax-Selbstspiel alle fünf
  Positionen berührt (3,1–3,7 s), 58 % erste Berührungen gegen den 2 Mrd. älteren Stand (Start 17 %),
  Anstoß-Tore 354:126 gegen den Start, Gesamtstärke nicht schlechter (+0,14 gegen den Start, +0,22 gegen
  das Referenz-Ende).
* **Grenzen:** Der Anstoß ist nicht schnell. In den Fünfteln 3–4 lag er bei 3,1 s mit 34–35 Boost und
  1500–1535 uu/s, im letzten Fünftel wieder bei 3,9 s mit 24 Boost und 1182 uu/s; argmax fährt weiter fast
  ohne Boost. Ein Speedflip (1,9–2,5 s) bleibt weit weg. Nebenwirkungen von `save_boost` 0,1: der
  Boost-Vorrat im Spiel sinkt von 0,43 auf ~0,36, die Entropie steigt auf 3,04–3,08 (Beobachtungsmarke
  3,0, §10.10). Ein Lauf je Variante; das Trainingsrauschen ist nur über die Referenz bekannt.
* Luftspiel: von keinem der Anstoß-Hebel berührt (0,68–0,75 Aerials je Minute).

**Vorschlag (nicht gestartet):** `train/configs/lucy_1v1_zero_sum_drill_fast_kickoff.json` = laufende
Hauptlauf-Config plus `save_boost` 0,1 und `kickoff_first_touch` 2,0 (Test
`test_kickoff_main_run_proposal_adds_only_the_two_kickoff_values`). Nach einem Wechsel beobachten: Anstoß
(unberührt, Zeit, Boost), `boost_held`, Entropie, und nach ~1 Mrd. Steps der Regressions-Check; die 200
Checkpoints erlauben den Rückweg. Der Hauptlauf läuft seit 02.10. 08:00 mit unveränderter Config ab
17.395.505.920 weiter.

### 11.6 Umstellung des Hauptlaufs und Beobachtung (02.10.2026, vom Nutzer freigegeben)

Am 02.10. um 09:35 sauber gestoppt (End-Checkpoint 18.527.085.056) und mit
`train/configs/lucy_1v1_zero_sum_drill_fast_kickoff.json` neu gestartet (`save_boost` 0,1,
`kickoff_first_touch` 2,0; beide Werte stehen so im Trainer-Log). Vorher lief der Hauptlauf von 08:00 bis
09:35 mit der alten Config von 17.395.505.920 bis 18.527.085.056.

Vorbemerkung: In der letzten Milliarde Steps vor dem Wechsel war der Anstoß unter der alten Config von
selbst wieder aktiv (unberührt 0,3 %, erste Berührung 3,0 s). Er pendelt dort zwischen aktiv und passiv
(§11.2: 26–36 % unberührt bei 16–17 Mrd.).

Verlauf (Trainingsmetriken, Mittel je 100 Mio. Steps seit dem Wechsel; Kontrollen alle ~15 min, alle
Grenzwerte aus §10 bei jeder Kontrolle eingehalten, SPS 188.000–197.000, keine Warnungen):

| Steps seit Wechsel | unberührt | erste Berührung | Tempo | Boost | `boost_held` | Entropie | Aerials/min | Tore/min | Value Loss | KL | Clip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 Mrd. davor | 0,3 % | 3,01 s | 1514 | 20,7 | 0,432 | 2,967 | 0,77 | 1,39 | 0,166 | 0,0047 | 0,038 |
| 0–100 Mio. | 0,0 % | 2,80 s | 1618 | 26,6 | 0,378 | 3,031 | 0,77 | 1,41 | 0,156 | 0,0046 | 0,037 |
| 200–300 Mio. | 0,0 % | 2,57 s | 1773 | 29,1 | 0,354 | 3,063 | 0,71 | 1,41 | 0,158 | 0,0046 | 0,036 |
| 500–600 Mio. | 0,0 % | 2,49 s | 1832 | 30,1 | 0,346 | 3,054 | 0,71 | 1,44 | 0,153 | 0,0046 | 0,036 |
| 800–900 Mio. | 0,0 % | 2,46 s | 1853 | 30,4 | 0,346 | 3,052 | 0,74 | 1,44 | 0,151 | 0,0046 | 0,036 |
| 900–1000 Mio. | 0,0 % | 2,48 s | 1829 | 29,6 | 0,346 | 3,053 | 0,71 | 1,43 | 0,153 | 0,0045 | 0,036 |

* Der Anstoß wird in den ersten ~500 Mio. Steps stetig schneller und bleibt dann bei ~2,5 s, ~1830 uu/s
  und ~30 Boost; kein Anstoß bleibt unberührt. Das ist schneller als der bisherige Bestwert (~2,6 s bei 7 Mrd., §10.10).
* `boost_held` fällt in den ersten 200 Mio. Steps von 0,43 auf ~0,35 und bleibt dort; die Entropie steigt
  von 2,97 auf ~3,05 und bleibt dort. KL, Clip und Value Loss ruhig; kein Sprung beim Wechsel.
* Aerials je Minute schwanken zwischen 0,68 und 0,75 (vorher 0,77), ohne Trend.

Regressions-Check nach 1 Mrd. Steps (11:03, 1000 Spiele): 19.528.093.568 gegen 18.527.085.056 (Stand beim
Wechsel) **gleich**, +0,00 [−0,16; +0,16] Tore/Spiel, Siege 401:437 (162 remis). Im Duell: Anstoß zuerst
84,9 % (Anstoß-Auswertung 86,0 % [82,7; 88,8]), Tempo bei der Berührung 2002 gegen 1730 uu/s, Boost 37,5
gegen 27,5, Tore binnen 10 s nach Anstoß 474:258. Außerhalb der Anstöße liegt der neue Stand also um etwa
0,2 Tore/Spiel zurück (Schüsse/min 1,30 gegen 0,99, aber Konversion im Angriffsdrittel 0,236 gegen 0,257,
Aerials 0,70 gegen 0,75).

Einordnung: Das Ziel der Umstellung ist erreicht, der Anstoß ist aktiv, stabil und klar besser als vor dem
Wechsel. Die Gesamtstärke hat in dieser Milliarde nicht zugelegt (davor +0,19 und +0,69 je 1–2 Mrd.); ob das
die Umgewöhnung an den kleineren Boost-Vorrat ist oder ein dauerhafter Preis von `save_boost` 0,1, zeigt
erst der nächste Check. Rückweg bei Bedarf: stoppen und `start_main_run.ps1` ohne `-Config`.

**Nachtrag nach 2 Mrd. Steps (02.10.2026, 12:30–13:15).** Alle Grenzwerte weiter eingehalten (20,53 Mrd.
Steps um 12:30, ~195.000 SPS). Trainingsmetriken je 250 Mio. Steps seit dem Wechsel: Anstoß ab 500 Mio.
unverändert bei 2,47 s, 1820–1864 uu/s, ~30 Boost, nie unberührt; `boost_held` 0,34–0,35; Entropie
3,02–3,05; Tore/min 1,39 → 1,46–1,50. Aerials je Minute sinken langsam: 0,77 (davor) → 0,70–0,73 (bis
1,25 Mrd.) → 0,68 (1,25–2 Mrd.).

| neu gegen alt | Urteil | Tordifferenz/Spiel | Siege | Anstoß: neu zuerst | Anstoß-Tore 10 s |
|---|---|---|---|---|---|
| 19.528.093.568 gegen 18.527.085.056 (Wechsel) | gleich | +0,00 [−0,16; +0,16] | 401:437 | 86,0 % | 474:258 |
| 20.529.058.304 gegen 19.528.093.568 | gleich | −0,14 [−0,31; +0,03] | 399:446 | 31,8 % | 231:173 |
| 20.729.243.520 gegen 18.727.286.656 (200 Mio. nach dem Wechsel) | **besser** | +0,70 [+0,54; +0,87] | 518:325 | 74,2 % | 353:215 |

Die drei Checks passen nicht transitiv zusammen (0,00 und −0,14 je Milliarde, aber +0,70 über zwei); einzelne
Checkpoints schwanken weiter um mehrere Zehntel Tore/Spiel (§11.3). Zusammen: kein Rückschritt durch die
Umstellung, über 2 Mrd. Steps ein Fortschritt in der Größe der letzten Checks vor dem Wechsel (+0,69 für
17,10 gegen 15,10 Mrd.), bei jetzt stabilem, schnellerem Anstoß. Offen: der langsame Rückgang der Aerials
(vermutlich der kleinere Boost-Vorrat) und weiterhin das Luftspiel insgesamt (§11.4, Punkt 3).

### 11.7 Höhepunkt bei ~28,9 Mrd., danach Rückgang; Neustart mit halbierter Lernrate (03.10.2026)

Der Hauptlauf lief auf der Anstoß-Config ohne Unterbrechung von 18,53 bis 39,56 Mrd. Steps (02.10. 09:35 bis
03.10. ~16:15), alle Grenzwerte eingehalten, ~195.000 SPS. Trainingsmetriken je 2 Mrd. Steps: Anstoß
durchgehend 2,45–2,51 s, nie unberührt; `boost_held` 0,34; Entropie 3,01–3,04; Aerials 0,71–0,75 je Minute;
Skill-Rating (gegen die letzten ~5 Mrd.) ab 18 Mrd. flach bei 3060–3100. Einziger Trend: KL 0,0046 (bis
22 Mrd.) → 0,0053 (26–28) → 0,0060 (32–34) → 0,0065–0,0068 (36–40), bei gleicher Lernrate und gleichem
Clip-Anteil (~3,7 %).

Regressions-Checks am 03.10.: 38,45 gegen 37,45 Mrd. **gleich** (+0,03 [−0,12; +0,19]); 38,45 gegen 28,49 Mrd.
**schlechter** (−0,44 [−0,61; −0,28], Siege 357:512, Anstoß zuerst 0,6 %, Schüsse/min 1,09 gegen 2,04).

Panel gegen einen festen Gegner (18.527.085.056 = Stand beim Wechsel, je 1000 Spiele, `duel.exe`,
`results/panel_vs_18527085056`, 38,45 in `results/duel_38446869376_vs_18527085056`):

| Stand | Tordifferenz/Spiel [95-%-KI] | Siege |
|---|---|---|
| 20.529.058.304 | −0,10 [−0,27; +0,06] | 405 |
| 28.487.006.720 | +0,75 [+0,58; +0,92] | 532 |
| 28.887.394.048 | **+0,92** [+0,76; +1,08] | 532 |
| 29.938.404.736 | +0,67 [+0,51; +0,83] | 515 |
| 31.189.640.704 | +0,58 [+0,42; +0,74] | 506 |
| 33.692.126.080 | +0,56 [+0,39; +0,73] | 512 |
| 36.194.602.112 | +0,36 [+0,19; +0,52] | 475 |
| 38.446.869.376 | −0,05 [−0,21; +0,12] | 402 |

Lesart: Nach dem Wechsel wurde der Lauf ~10 Mrd. Steps lang deutlich stärker (Höhepunkt um 28,9 Mrd.) und
fiel danach über 10 Mrd. Steps gleichmäßig auf das Niveau vom Wechsel zurück. Der Anstoß bleibt dabei gut;
der Verlust liegt im übrigen Spiel. Die 1-Mrd.-Checks und das Skill-Rating sehen das nicht, weil sie nur
gegen nahe Vorgänger messen, die gleich mitgefallen sind. Die steigende KL bei konstanter Lernrate passt zu
einem Lauf, der spät mit zu großen Schritten hin- und herpendelt, statt sich zu verfeinern; bewiesen ist die
Ursache nicht (reines Selbstspiel kann auch zyklisch vergessen).

Sicherung: Die vollständigen Checkpoints 28.887.394.048, 29.938.404.736, 31.189.640.704 und 33.692.126.080
wurden vor der Rotation nach `runs/backup_lucy_1v1/` kopiert (Hash geprüft); 28.487.006.720 war schon
rotiert (nur `PPO_POLICY.lt` in `results/regression/` erhalten).

**Neustart (vom Nutzer freigegeben):** Hauptlauf am 03.10. sauber gestoppt (End-Checkpoint 39.558.949.632 in
`runs/lucy_1v1`, dort nichts verändert) und um 16:32 mit `train/configs/lucy_1v1_kickoff_lr1e4.json`
gestartet: Anstoß-Config mit `policy_lr` und `critic_lr` 2e-4 → 1e-4, eigener Ordner
`runs/lucy_1v1_lr1e4/checkpoints` ab einer geprüften Kopie von 28.887.394.048 (stärkster gemessener Stand).
Der Trainer setzt die Lernrate nach dem Laden aus der Config (`PPOLearner::LoadFrom`, Log „Updated learning
rate to [1e-04, 1e-04]“). Erste Iterationen: ~193.000 SPS, KL 0,0034, Clip 2,4 %, Grenzwerte eingehalten.
Ab jetzt gelten Status und Regressions-Check mit `--run runs\lucy_1v1_lr1e4`.

**Ergebnis nach 2 Mrd. Steps mit Lernrate 1e-4 (03.10.2026, 19:20–19:36).** 30.889.387.904 (je 1000 Spiele,
`results/lr1e4_check_30889387904`):

| neu gegen | Tordifferenz/Spiel [95-%-KI] | Anstoß zuerst | Schüsse/min | Aerials/min |
|---|---|---|---|---|
| festen Gegner 18,53 Mrd. | **+1,55** [+1,38; +1,71] (Start 28,89: +0,92; alter Lauf bei 29,94/31,19: +0,67/+0,58) | 91 % | 1,83 : 0,86 | 0,86 : 0,65 |
| den Start 28,89 Mrd. | **+0,72** [+0,55; +0,89] | 70 % | 1,52 : 1,12 | 0,86 : 0,73 |

Vom selben Checkpoint aus fiel der Lauf mit 2e-4 in den folgenden 2,3 Mrd. Steps auf +0,58 gegen den festen
Gegner, mit 1e-4 stieg er auf +1,55. Trainingsmetriken je 500 Mio. Steps: KL 0,0036 → 0,0041, Clip 2,4–2,6 %,
Entropie 2,98–3,00, Anstoß 2,43–2,51 s und nie unberührt, `boost_held` 0,34; Aerials je Minute 0,77 → 0,85
(alter Lauf zuletzt 0,71–0,75); Skill-Rating 3049 → 3107. Ein Lauf, aber ein großer Effekt; KL steigt
langsam wieder und bleibt im Blick.

**Panel nach 14,3 Mrd. Steps mit Lernrate 1e-4 (04.10.2026, 12:25–13:04).** Je 1000 Spiele gegen den festen
Gegner 18,53 Mrd. (`results/panel2_vs_18527085056`):

| Stand | Tordifferenz/Spiel [95-%-KI] | Anstoß zuerst | Aerials/min (Stand : Gegner) |
|---|---|---|---|
| 30.889.387.904 | +1,55 [+1,38; +1,71] | 91 % | 0,86 : 0,65 |
| 33.191.647.744 | +2,00 [+1,84; +2,17] | 88 % | 0,93 : 0,67 |
| 35.193.669.632 | +1,90 [+1,72; +2,07] | 80 % | 0,89 : 0,68 |
| 37.195.705.088 | +2,41 [+2,23; +2,58] | 84 % | 0,92 : 0,66 |
| 39.197.720.448 | +2,30 [+2,13; +2,46] | 84 % | 0,94 : 0,68 |
| 41.199.756.928 | **+2,51** [+2,34; +2,68] | 89 % | 0,97 : 0,64 |
| 43.151.733.632 | +2,38 [+2,22; +2,55] | 90 % | 0,90 : 0,64 |

Kein Rückgang wie mit 2e-4: Anstieg bis ~37 Mrd., seitdem ein Plateau bei +2,3 bis +2,5 (die letzten drei
Werte liegen im Rauschen beieinander). Trainingsmetriken je 2 Mrd.: Skill-Rating 3060 → 3168 (10–12 Mrd. nach
dem Neustart) → 3146; Aerials 0,80 → 0,93 → 0,88; KL 0,0038 → 0,0049, Clip 2,4 → 2,8 %. Gesichert in
`runs/backup_lucy_1v1_lr1e4/`: 37.195.705.088 und 41.199.756.928 (Hash geprüft). Vorschlag: diesen Vergleich
etwa alle 5 Mrd. Steps wiederholen und den besten Stand sichern; steigt die KL weiter Richtung 0,006, eine
weitere Lernraten-Stufe prüfen.

### 11.8 Luftspiel: Reward für Luftkontakte und mehr Aerial-Startlagen (04.10.2026, vom Nutzer freigegeben)

Drei Läufe zu je 1 Mrd. Steps ab 41.199.756.928 (Kopie in `runs/backup_lucy_1v1_lr1e4`), Seed 123, Build `1cd931b`,
nacheinander als Aufgabe „RLbot Experimente“ (`results/sp3_chain.ps1`, Hauptlauf dafür von 13:40 bis 19:26
gestoppt, die Kette hat ihn am Ende selbst mit unveränderter Config wieder gestartet). Referenz = laufende
Hauptlauf-Config (`sp3_reference`); `sp3_air_touch`: `air_touch` 3,0 (Luftkontakt, skaliert mit der Ballhöhe);
`sp3_air_touch_aerial`: dazu `state_setters.aerial` 0,5 → 2,0. Checkpoints alle 100 Mio.; die Stände bei
600/800/1000 Mio. spielten je 1000 Spiele gegen den festen Gegner 18.527.085.056 (`results/sp3_panel.md`),
dazu `compare.py` (`results/compare_sp3.md`). Vorab: `air_touch` macht beim aktuellen Bot ~0,4 % des
Reward-Flusses aus (`reward_budget.exe`, 60 Spiele), weil hohe Kontakte selten sind.

| | Referenz | `air_touch` 3,0 | `air_touch` 3,0 + Startlagen |
|---|---|---|---|
| Panel gegen festen Gegner, 600 / 800 / 1000 Mio. | +1,95 / +2,21 / +2,17 | +1,96 / +1,82 / +1,70 | +2,38 / **+2,70** / +2,29 |
| Panel, Mittel | +2,11 | +1,82 | **+2,46** |
| Panel, Aerials je Minute (Mittel) | 0,95 | 0,99 | 1,01 |
| Panel, mittlere Höhe der Luftkontakte | 218 uu | 220 uu | 221 uu |
| Duell Ende gegen Referenz-Ende | – | +0,26 [+0,09; +0,42] | −0,27 [−0,44; −0,10] |
| Duell Ende gegen den eigenen Start | +0,03 [−0,14; +0,19] | +0,19 [+0,02; +0,37] | −0,17 [−0,33; +0,00] |
| TrueSkill gemeinsame Ladder (mu; Start 25,12) | 24,98 | 25,08 | 24,87 |
| Training: Aerials je Minute, erstes → letztes Fünftel | 0,96 → 0,89 | 0,96 → 0,96 | 0,98 → 1,05 |
| Training: mittlere Höhe der Luftkontakte (letztes Fünftel) | 192 uu | 195 uu | 204 uu |

Lesart:

* **Kein großer Effekt in 1 Mrd. Steps.** Gegen den festen Gegner haben alle drei Läufe ~0,95–1,01 Aerials je
  Minute bei ~220 uu Höhe; der Unterschied zur Referenz ist +4 bis +6 %. Im Training steigt die Aerial-Rate nur
  mit beiden Änderungen (0,98 → 1,05, Referenz 0,96 → 0,89); ein Teil davon ist die andere Szenen-Mischung.
* **Die Stärke-Vergleiche widersprechen sich.** `air_touch` allein schlägt das Referenz-Ende direkt (+0,26), liegt
  gegen den festen Gegner aber an zwei von drei Zeitpunkten unter der Referenz. Das Bündel ist gegen den festen
  Gegner an allen drei Zeitpunkten vorn (Mittel +2,46 gegen +2,11), verliert aber das direkte Duell gegen das
  Referenz-Ende (−0,27). Die Ladder sieht alle vier Stände im Rauschen (mu 24,87–25,12, σ 0,79). Das passt zur
  bekannten Nicht-Transitivität einzelner Checkpoints (§11.3, §11.7); dazu passt auch, dass beide Experimente
  gegen das Referenz-Ende nur 25 % der Anstöße zuerst berühren.
* Nebenwirkungen: keine (Anstoß 2,43–2,45 s, `boost_held` 0,35, KL ~0,005, Entropie ~2,95, ~199.000 SPS).

Einordnung: Mehr Luftspiel kam in diesem Projekt bisher vor allem aus langem Training mit kleinerer Lernrate
(0,77 → ~0,95 Aerials je Minute über 14 Mrd. Steps, §11.7). Die beiden Hebel ändern das in 1 Mrd. Steps nur
wenig. Größere Hebel (Ballvorhersage in der Beobachtung, größeres Netz, schnellerer Entscheidungstakt) brechen
die Checkpoint-Kompatibilität.

**Vorschlag (nicht gestartet):** `train/configs/lucy_1v1_kickoff_lr1e4_air.json` = laufende Hauptlauf-Config plus
das Bündel (`air_touch` 3,0, `aerial` 2,0); bester Wert gegen den festen Gegner und einziger steigender
Aerial-Trend, Nachteil im direkten Duell im Rahmen des Checkpoint-Rauschens. Nach einem Wechsel: Panel gegen
den festen Gegner alle ~5 Mrd. Steps; fällt der Lauf darin unter die Linie der Referenz, zurück zur bisherigen
Config. Der Hauptlauf läuft seit 04.10. 19:26 unverändert ab 43.706.476.800.
