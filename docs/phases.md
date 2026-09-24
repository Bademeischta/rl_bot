# Phasen: Stand der Umsetzung

Alles hier Beschriebene ist gebaut und getestet, sofern nicht ausdrücklich anders vermerkt.
Die Messwerte aus Phase 0 stehen in [phase0_results.md](phase0_results.md).

## Überblick

| Phase | Inhalt | Stand |
|---|---|---|
| 0 | Hardware-Audit, SPS-Messung, Toolchain-Spike | **fertig**, Ergebnis: C++ auf GPU, 117k SPS |
| 1 | Sanity-Lauf | **fertig**, 20 Mio. Steps, Lernsignal bestätigt |
| 2 | Lucy-nahe Rewards, Obs mit k=5, State-Setter | **fertig**, 39 C++-Tests |
| 3 | Hauptlauf 1v1 | **läuft** (rechenzeitgebunden, mehrere Tage) |
| 4 | Multi-Mode 1v1/2v2/3v3 | **Werkzeuge fertig**, Config `lucy_multimode.json`, Start nach Phase 3 |
| 5 | Feinschliff, Liga/TrueSkill | **Werkzeuge fertig** (`duel.exe`, `eval/ladder.py`) |
| 6 | Deployment über RLBot v5 | **fertig**, Start erst mit brauchbarem Checkpoint sinnvoll |

## Phase 1: Sanity

```powershell
.\build\cpp_cu128\train_bot.exe train\configs\sanity.json
.\.venv\Scripts\python tools\show_metrics.py runs\sanity\metrics.csv
```

Ergebnis des Laufs vom 23.09.2026 (20 Mio. Steps, ~4 Minuten):

| Steps | Episoden-Reward | Entropie | in-air-Anteil | Ballkontaktrate |
|---|---|---|---|---|
| 0,1 Mio. | −5,2 | 4,499 | 0,68 | 0,00010 |
| 9,9 Mio. | +11,0 | 4,317 | 0,62 | 0,00029 |
| 16,3 Mio. | +58,3 | 3,968 | 0,11 | 0,00042 |

Das ist das erwartete frühe Bild: Die Policy verlässt das Zufallsverhalten (Entropie sinkt),
hört auf sinnlos herumzufliegen und sammelt langsam mehr Ballkontakte.

## Phase 2: Lucy-nahe Baseline

**Rewards** (`env/cpp/Rewards.h`), alle einzeln gegen von Hand nachgerechnete Werte getestet:
- `KRCReward`: vorzeichenbehaftetes geometrisches Mittel (arXiv 2305.15801, eq. 2). Eine
  schlechte Komponente kann nicht durch eine gute überdeckt werden; ist eine Komponente 0,
  ist das Ergebnis 0.
- `ParamDistanceReward`: `exp(-0.5·d/(c·w_dis))^(1/w_den)` (eq. 3).
- `AlignBallGoalReward`, `TouchBallToGoalAccelReward`, `InAirReward`, dazu die Upstream-Rewards
  `EventReward` (Tor/Gegentor/Demo), `SaveBoostReward` (Wurzel), `VelocityPlayerToBallReward`.
- Zusammengesetzt: „Offensive Potential" = KRC(Align, VelocityToBall, Distance),
  „Distance-weighted Alignment" = KRC(Align, Distance).
- `team_spirit` > 0 schaltet über `ZeroSumReward` auf zero-sum mit Teamverteilung.

**Beobachtung** (`env/cpp/Obs.h`), 257 Werte bei max. 3v3:
Ball (9) + Boost-Pad-Timer (34) + eigener Block (29) + 5 gestapelte Aktionen (40)
+ 2 Mitspieler-Slots (58) + 3 Gegner-Slots (87). Slots werden im Training gemischt und
mit Nullen aufgefüllt, dadurch ist dasselbe Netz für 1v1, 2v2 und 3v3 verwendbar.

**State-Setter** (`env/cpp/StateSetters.h`): Kickoff, Random, Aerial, Dribble, Wall-Play,
Recovery, Defense, gewichtet auswählbar. Das ist das Mechanik-Curriculum aus dem Bauplan.

**Nicht umgesetzt:** die Aux-Tasks aus Lucy-SKG (State Representation, Reward Prediction).
Die brauchen zusätzliche Köpfe und Loss-Terme im PPO-Learner, also einen Eingriff in
RLGymPPO_CPP. Das ist bewusst offen gelassen, siehe „Offene Punkte".

## Phase 3: Hauptlauf 1v1

```powershell
.\build\cpp_cu128\train_bot.exe train\configs\lucy_1v1.json
```

Der Lauf ist unbegrenzt (`timestep_limit: 0`) und schreibt alle 25 Mio. Steps einen Checkpoint
nach `runs/lucy_1v1/checkpoints/<steps>/`.

### Pausieren und Fortsetzen

Pausieren (am besten kurz nach einem Checkpoint, sonst gehen bis zu 25 Mio. Steps
verloren — bei aktuellem Tempo rund 11 Minuten):

```powershell
Get-Process train_bot | Stop-Process -Force
```

Fortsetzen: **derselbe Befehl wie beim Start.** Der Learner sucht im Checkpoint-Ordner
automatisch den höchsten Step-Stand und lädt ihn:

```powershell
.\build\cpp_cu128\train_bot.exe train\configs\lucy_1v1.json
```

Im Log steht dann `Loading most recent checkpoint in "runs/lucy_1v1/checkpoints"...`.
Geprüft am 23.09.2026: Nach dem Stopp bei 125.222.784 Steps lief der Neustart bei exakt
diesem Stand weiter, Modell-Updates (7302 → 7344) und Skill-Rating (1122,7) wurden übernommen.
Die Metriken in `metrics.csv` werden fortgeschrieben, nicht überschrieben.

Für einen Dauerlauf, der auch nach dem Schließen des Terminals weiterläuft:

```powershell
Start-Process .\build\cpp_cu128\train_bot.exe -ArgumentList "train\configs\lucy_1v1.json" -RedirectStandardOutput runs\lucy_1v1.log -WindowStyle Hidden
```

Reward-Phasen B und C (Bauplan §6) werden **nicht** nach Step-Zahl umgestellt, sondern erst
bei einem TrueSkill-Plateau: `rewards.velocity_player_to_ball` und `save_boost` senken,
danach die dichten Terme weiter Richtung 0.

Gemessener Durchsatz mit dieser Config (Netz 512×3, 16 Threads × 64 Spiele):
**rund 68.000 SPS im eingeschwungenen Zustand, also 5,9 Mrd. Steps pro Tag** und gut 5 Tage
für 30 Mrd. Steps. Die ersten ~180 Mio. Steps laufen nur mit ~47.000 SPS, weil die Policy den
Ball noch kaum trifft und die Episoden deshalb ständig in den NoTouch-Timeout laufen.
Nachrechnen mit `python tools/throughput.py runs/lucy_1v1`, Verlauf mit
`python tools/timing_trend.py runs/lucy_1v1`, Aufteilung vergleichen mit
`python tools/tune_threads.py`.

Ein erster Anlauf mit dem größeren Netz 1024/1024/512/512 lief 4,1 Stunden und kam auf
429 Mio. Steps (28.762 effektive SPS, also halbes Tempo). Er liegt unter
`runs/archive/lucy_1v1_net1024/`; die Checkpoints sind wegen der anderen Netzgröße nicht
mit dem aktuellen Lauf kompatibel. Das Verhalten war unauffällig: Ballkontaktrate 16× gestiegen,
Entropie stabil bei ~3,2, Skill-Rating monoton 1000 → 1563.

## Phase 4: Multi-Mode

`train/configs/lucy_multimode.json` verteilt die Environments nach `env.mode_mix` auf 1v1/2v2/3v3.
Die Verteilung nutzt das Größte-Reste-Verfahren, trifft die Gewichte also exakt und bringt auch
kleine Anteile von Anfang an unter (getestet in `tests/cpp/test_config.cpp`).

Der Modus-Mix und `team_spirit` sollen laut Bauplan an der Eval-Metrik hängen, nicht an einem
festen Plan: stagniert die 2v2-Wertung, Anteil erhöhen.

## Phase 5: Eval und Liga

```powershell
# Zwei Checkpoints direkt gegeneinander
.\build\cpp_cu128\duel.exe --a <A>\PPO_POLICY.lt --b <B>\PPO_POLICY.lt --games 100

# Neuester Checkpoint gegen ältere, mit TrueSkill-Tabelle
.\.venv\Scripts\python eval\ladder.py --run runs\lucy_1v1 --games 100
```

- Die Seiten werden jedes Spiel getauscht, damit Kickoff- und Seitenvorteile sich aufheben.
  Verifiziert über das Defense-Szenario: Dort trifft der Ball immer das blaue Tor, das Ergebnis
  muss deshalb exakt ausgeglichen sein — gemessen 15:15 in 30 Spielen.
- Bewertet wird **pro Tor**, nicht pro Spiel (geringere Varianz, wie bei Lucy-SKG).
- TrueSkill-Parameter wie in der Seer-Arbeit (mu=25, sigma=mu/3, beta=sigma/2, tau=sigma/100),
  ausgewiesen wird `mu − 3·sigma`.
- Statistik: `standard_error(p, n)`; für ±5 % braucht es rund 100 Spiele, für ±2 % rund 625.

Gegen Necto/Nexto zu messen ist **nicht** Teil davon: Das bräuchte RLBot mit dem Botpack und
ein laufendes Spiel, geht also nicht headless in RocketSim.

## Phase 6: Deployment

```powershell
.\.venv\Scripts\python tools\export_policy.py runs\lucy_1v1\checkpoints --out deploy\rlbot\policy.pt
```

Danach `deploy/rlbot/bot.toml` im RLBot-Launcher laden. **Nur offline und ohne EAC**
(„Launch without EAC"), nie online oder ranked.

Die drei Stellen, an denen Training und Deployment auseinanderlaufen könnten, sind jeweils mit
Golden-Tests gegen den C++-Code abgesichert:

| Risiko | Absicherung | Ergebnis |
|---|---|---|
| Obs-Layout weicht ab | `tests/test_obs_parity.py` gegen `dump_obs.exe` | 120 Vektoren, Abweichung 0 |
| Aktionstabelle verschoben | `tests/test_action_table_parity.py` | 90 Aktionen identisch |
| Rotation falsch umgerechnet | `tests/test_rotation_parity.py` | 47 Fälle, Abweichung < 1e-5 |
| Inferenz rechnet anders | `tests/test_policy_parity.py` gegen `dump_policy_actions.exe` | 120 Obs, Differenz 0,0 |

Dazu die Fallstricke der RLBot-API, die in `deploy/packet_adapter.py` gekapselt und getestet sind:
- RLBots Boost-Timer zählt **seit** dem Aufsammeln hoch, RocketSim zählt die Restzeit herunter.
- Die Pad-Reihenfolge von RLBot ist nicht die von RLGym; zugeordnet wird über die Positionen.
- `dodge_timeout` ist −1 sowohl am Boden als auch nach Ablauf des Flip-Fensters; beide Fälle
  müssen getrennt werden, sonst hätte ein stehendes Auto laut Obs keinen Flip mehr.
- Der Bot entscheidet nur alle 8 Ticks neu, genau wie im Training (15 Entscheidungen/s).

## Offene Punkte

- **Aux-Tasks (Lucy SR/RP):** nicht umgesetzt, würde Eingriffe in den PPO-Learner brauchen.
  Laut Paper der stärkere Hebel für Sample-Effizienz (RP-Plateau bei 100 Mio. statt 200 Mio. Steps).
- **PRISM-Bausteine:** Der State-Setter-Teil (Punkt 5) ist da. Regret-Prioritized State Buffer,
  Dual-Anchor und Symmetry-Critic sind nicht umgesetzt; sie waren im Bauplan als
  „unbelegt, braucht Spike" markiert und gehören hinter eine funktionierende Baseline.
- **BC-Prior aus Replays:** nicht umgesetzt (eigenes Arbeitspaket, ~20–40 Std.).
- **Vergleich mit Nexto/Necto:** braucht RLBot mit Botpack und ein laufendes Spiel.
- **Stromaufnahme:** nicht gemessen.
