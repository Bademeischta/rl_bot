# Stufe-3-Experimente (AUDIT.md Roadmap, Punkte 11–14)

Jedes Experiment startet vom **selben Checkpoint**, mit dem **selben Seed** und der **selben
Step-Zahl** wie der Kontrolllauf, und ändert gegenüber `baseline.json` genau **eine** Sache
(geprüft in `tests/test_experiment_configs.py`).

| Config (`train/configs/experiments/`) | Änderung | Audit |
|---|---|---|
| `baseline.json` | keine (= `lucy_1v1.json` nach K1a/K1b, plus `env.seed_envs: true` wie in allen Experiment-Configs; Default ist false, Review R6) | Kontrolllauf |
| `h2_ent_coef_0004.json` | `learner.ent_coef` 0,01 → 0,004 | H2 |
| `h3_no_shuffle.json` | `env.shuffle_slots` → false | H3 |
| `k3_rewards.json` | **Bündel aus 7 Werten** laut AUDIT.md K3 (goal/concede 50, vier dichte Terme runter, in_air 0). Ein Effekt ist keinem einzelnen Wert zuzuordnen; `compare.py` markiert das. Aufteilen nur, wenn das Ergebnis schlecht oder unklar ist (Review R11) | K3 |
| `zero_sum.json` (früher `team_spirit_01.json`) | `rewards.team_spirit` 0 → 0,1 **und** `goal`/`concede` 10 → 5. Im 1v1 ist τ wirkungslos: jedes τ > 0 ergibt `r_i − r_j` (Zero-Sum). Zero-Sum zählt Tore doppelt, deshalb halbiert: nach dem Wrapper bleibt ein Tor ±10 wie in der Baseline, gemessen wird nur der Zero-Sum-Effekt auf das Shaping. `Average Step/Episode Reward` sind hier konstant 0 → `raw_step_reward` (vor dem Wrapper) ansehen | optional, K3 zweiter Hebel (Review R10) |

`RUNNING_STATS.json` wird bei allen Experimenten **übernommen** (Begründung AUDIT.md §7.3).

## Reihenfolge lokal

```powershell
# 1. Kontrolllauf (Pflicht, zuerst)
powershell -ExecutionPolicy Bypass -File tools\experiments\run_experiment.ps1 `
    -Config train\configs\experiments\baseline.json `
    -StartCheckpoint runs\lucy_1v1\checkpoints\2704829056 -Steps 100000000 -Seed 123

# 2. Experimente, jeweils mit -Baseline auf den Ergebnisordner des Kontrolllaufs
powershell -ExecutionPolicy Bypass -File tools\experiments\run_experiment.ps1 `
    -Config train\configs\experiments\h2_ent_coef_0004.json `
    -StartCheckpoint runs\lucy_1v1\checkpoints\2704829056 -Steps 100000000 -Seed 123 `
    -Baseline results\exp_baseline_<datum>
# ... dito h3_no_shuffle, k3_rewards, zero_sum

# 3. Vergleich
.\.venv\Scripts\python tools\experiments\compare.py results\exp_* --out results\compare.md
```

Dauer pro Lauf: 100 Mio. Steps ÷ lokale SPS. Bei den im Audit gelesenen ~68.000 SPS wären das
rund 25 Minuten plus Ladder/Duelle (wenige Minuten); die echte Dauer steht in `summary.json`
(`wall_seconds`) — **lokal nachmessen**, siehe AUDIT.md §0.

Was `run_experiment.ps1` macht: Speicherplatz prüfen → Checkpoint (inkl. `RUNNING_STATS.json`)
nach `runs\exp_<name>_<datum>\checkpoints\` **kopieren** (Original unberührt, vorhandene Ordner
werden nie überschrieben) → Training mit `learner.extra_steps` und `save_on_exit` → Ladder
(`eval\ladder.py`, TrueSkill, `ratings.json` im Lauf-Ordner) und Duelle Ende-gegen-Start sowie
Ende-gegen-Baseline-Ende (`duel.exe`, 100 Spiele) → `results\exp_<name>_<datum>\` mit
`summary.md`/`summary.json`, `metrics.csv`, Ladder, Duellen, Logs → `results\exp_<name>_<datum>.zip`.

## Abbruchkriterien (`check_abort.py`, alle 30 s auf `metrics.csv`)

Die Schwellen sind bewusst weit: Ein Lauf kostet 25–40 Minuten, ein Fehlalarm kostet mehr als
ein schlechter Lauf. Alle Fenster sind Iterationen (1 Iteration ≈ 100.000 Steps).

| Kriterium | Schwelle | Begründung |
|---|---|---|
| `nan`, `inf` oder leeres Feld in Entropie, Value Loss, KL, Advantage, Val Target | sofort | Einmal vergiftet, ist der Rest des Laufs wertlos; kein legitimer Zustand erzeugt NaN. Der Trainer schreibt nan/inf wörtlich, ein leeres Feld heißt „Schlüssel fehlte" (Review R5). **Nicht** dabei: `Average Episode Reward` – dort heißt nan nur „keine Episode beendet" (kommt nach Neustarts vor); `summary.md` zählt diese Iterationen |
| Explodierender Value Loss | Median der letzten 20 Iterationen > **10 ×** Median des Referenzfensters (Iteration 100–200 desselben Laufs) **und** absolut > 100; erst ab Iteration 220 | K3 lässt den Critic absichtlich neu einschwingen (Val Target ~10 → ~3), deshalb Aufwärmphase und Referenz aus dem eigenen Lauf statt aus der Baseline. Normale Drift über 1.000 Iterationen liegt weit unter Faktor 2; Faktor 10 ist Divergenz |
| SPS-Einbruch | Mittel „Overall Steps/Second" der letzten 20 Iterationen < **40 %** der Referenz (Iteration 20–120, oder Baseline-SPS) | Messstreuung bis 7 %, thermisches Drosseln rund 30 % (`docs/phase0_results.md`); unter 40 % läuft etwas anderes (CPU-Fallback, Fremdprozess, Swap) und das Step-Budget misst Unsinn |
| Entropie-Kollaps | Policy Entropy < **2,5** → nur **Warnung** | Schwelle aus AUDIT.md H2 („Weg zurück billig"); das Ergebnis bleibt auswertbar, deshalb kein Abbruch |

Rückgabewerte von `check_abort.py`: 0 weiter, 3 abbrechen, 4 Warnung.

## Vergleich (`compare.py`)

**Hauptkriterium** (Review R12): das Duell jedes Experiment-Endes gegen das Baseline-Ende
(`duel_end_vs_baseline.json`, 100 Spiele): Gewinnrate (Remis = halber Sieg) mit
95-%-Wilson-Intervall. Intervall ganz über 50 % = besser, ganz darunter = schlechter, sonst unklar.

**TrueSkill** nur aus **einer gemeinsamen Ladder**, die `compare.py` selbst mit `duel.exe` spielt:
alle Experiment-Enden plus Baseline-Start und Baseline-Ende, jeder gegen jeden
(`--ladder-games`, Default 50 je Paarung; 0 = aus). Ergebnis in `joint_ladder.json` neben `--out`.
Die Einzel-Ladders in den Lauf-Ordnern haben jeweils eigene Nullpunkte und werden für den
Vergleich nicht benutzt.

Dazu pro Experiment (letztes Fünftel der Iterationen): Tor-Anteil (`ep_end_goal`),
Timeout-Anteil, Episodenlänge, Ballkontakt, Entropie, Clip-Fraction, KL, Value Loss, Val Target,
Truncated Steps, SPS mit Differenz zur Baseline, der Toranteil im Duell (± SE) und die Liste der
Config-Änderungen (Bündel markiert, R11). Die Hinweise am Ende sind regelbasiert; die
Entscheidung behalten / verwerfen / nachmessen bleibt beim Menschen, Kriterien in AUDIT.md §6
Stufe 3.
