"""TrueSkill-Ladder für Checkpoints (Bauplan v2 §8).

    python eval/ladder.py --run runs/lucy_1v1 --games 50
    python eval/ladder.py --a runs/x/checkpoints/100 --b runs/x/checkpoints/200 --games 100

Bewertet wird pro Tor statt pro Spiel: Die Tordifferenz hat weniger Varianz als Sieg/Niederlage,
bei gleichem Rechenaufwand (Lucy-SKG wertet ebenfalls Einzeltor-Matches).
Parameter wie in der Seer-Arbeit: mu=25, sigma=mu/3, beta=sigma/2, tau=sigma/100.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import trueskill

ROOT = Path(__file__).resolve().parents[1]
DUEL_EXE = ROOT / "build" / "cpp_cu128" / "duel.exe"
RATINGS_PATH = ROOT / "eval" / "ratings.json"

MU = 25.0
SIGMA = MU / 3
ENV = trueskill.TrueSkill(mu=MU, sigma=SIGMA, beta=SIGMA / 2, tau=SIGMA / 100, draw_probability=0.0)


@dataclass
class DuelResult:
    name_a: str
    name_b: str
    games: int
    goals_a: int
    goals_b: int
    wins_a: int
    wins_b: int
    draws: int
    seconds: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def goal_share_a(self) -> float:
        total = self.goals_a + self.goals_b
        return 0.5 if total == 0 else self.goals_a / total


def standard_error(p: float, n: int) -> float:
    """Standardfehler eines Anteils. Für +-5 % bei p=0,5 braucht es rund 100 Spiele."""
    return math.sqrt(p * (1 - p) / n) if n > 0 else float("inf")


def games_needed(margin: float, p: float = 0.5) -> int:
    """Wie viele Spiele für einen gewünschten Standardfehler nötig sind."""
    return math.ceil(p * (1 - p) / (margin ** 2))


def run_duel(path_a: Path, path_b: Path, games: int, team_size: int = 1,
             deterministic: bool = False, setter: str = "kickoff",
             max_seconds: int = 120, exe: Path = DUEL_EXE) -> DuelResult:
    if not exe.exists():
        raise FileNotFoundError(f"duel.exe fehlt: {exe} (bench\\cpp\\build.ps1 -Target duel)")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        cmd = [str(exe), "--a", str(path_a), "--b", str(path_b), "--games", str(games),
               "--team-size", str(team_size), "--setter", setter,
               "--max-seconds", str(max_seconds), "--out", str(out)]
        if deterministic:
            cmd.append("--deterministic")
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if proc.returncode != 0:
            raise RuntimeError(f"duel.exe fehlgeschlagen:\n{proc.stdout}\n{proc.stderr}")
        data = json.loads(out.read_text(encoding="utf-8"))

    return DuelResult(
        name_a=path_a.parent.name if path_a.name.endswith(".lt") else path_a.name,
        name_b=path_b.parent.name if path_b.name.endswith(".lt") else path_b.name,
        games=data["games"], goals_a=data["goals_a"], goals_b=data["goals_b"],
        wins_a=data["wins_a"], wins_b=data["wins_b"], draws=data["draws"],
        seconds=data.get("seconds", 0.0), raw=data,
    )


def interleave_goals(goals_a: int, goals_b: int, a_first_on_tie: bool = True) -> list[bool]:
    """Verteilt die Tore gleichmäßig über die Sequenz (True = Tor für A).

    TrueSkill-Updates sind reihenfolgeabhängig: Erst alle Tore von A und dann alle von B zu
    werten, lässt den letzten Block dominieren (25:25 ergäbe so über 10 Punkte Abstand).
    Jedes Tor bekommt deshalb seinen Bruchrang (i+0.5)/n; nach diesem Rang sortiert liegen
    beide Serien gleichmäßig ineinander. `a_first_on_tie` entscheidet bei exakt gleichen
    Rängen und wird von update_ratings kanonisch gesetzt, damit die Namensreihenfolge
    das Ergebnis nicht verändert.
    """
    events: list[tuple[float, int, bool]] = []
    for i in range(goals_a):
        events.append(((i + 0.5) / goals_a, 0 if a_first_on_tie else 1, True))
    for j in range(goals_b):
        events.append(((j + 0.5) / goals_b, 1 if a_first_on_tie else 0, False))
    events.sort(key=lambda e: (e[0], e[1]))
    return [scored_by_a for _, _, scored_by_a in events]


def update_ratings(ratings: dict[str, trueskill.Rating], result: DuelResult
                   ) -> dict[str, trueskill.Rating]:
    """Ein Rating-Update pro Tor. Ein torloses Duell ändert nichts."""
    ratings = dict(ratings)
    for name in (result.name_a, result.name_b):
        ratings.setdefault(name, ENV.create_rating())

    # Kanonische Reihenfolge: mehr Tore zuerst, bei Gleichstand der kleinere Name.
    # Dadurch liefert ein Duell dasselbe Ergebnis, egal wer als A übergeben wird.
    if result.goals_a != result.goals_b:
        a_first = result.goals_a > result.goals_b
    else:
        a_first = result.name_a < result.name_b

    for a_scored in interleave_goals(result.goals_a, result.goals_b, a_first):
        first, second = ((result.name_a, result.name_b) if a_scored
                         else (result.name_b, result.name_a))
        winner, loser = trueskill.rate_1vs1(ratings[first], ratings[second], env=ENV)
        ratings[first], ratings[second] = winner, loser
    return ratings


def conservative(rating: trueskill.Rating) -> float:
    """mu - 3*sigma: der Wert, den man guten Gewissens behaupten kann."""
    return rating.mu - 3 * rating.sigma


def load_ratings(path: Path = RATINGS_PATH) -> dict[str, trueskill.Rating]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: ENV.create_rating(v["mu"], v["sigma"]) for k, v in data.items()}


def save_ratings(ratings: dict[str, trueskill.Rating], path: Path = RATINGS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {k: {"mu": r.mu, "sigma": r.sigma, "conservative": conservative(r)}
         for k, r in sorted(ratings.items(), key=lambda kv: -conservative(kv[1]))},
        indent=2), encoding="utf-8")


def checkpoints_of(run_dir: Path) -> list[Path]:
    folder = run_dir / "checkpoints" if (run_dir / "checkpoints").exists() else run_dir
    dirs = [d for d in folder.iterdir() if d.is_dir() and d.name.isdigit()
            and (d / "PPO_POLICY.lt").exists()]
    return sorted(dirs, key=lambda d: int(d.name))


def print_table(ratings: dict[str, trueskill.Rating]) -> None:
    print(f"\n{'Checkpoint':>16}  {'mu':>7}  {'sigma':>6}  {'mu-3sigma':>10}")
    for name, r in sorted(ratings.items(), key=lambda kv: -conservative(kv[1])):
        print(f"{name:>16}  {r.mu:>7.2f}  {r.sigma:>6.2f}  {conservative(r):>10.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, help="Run-Ordner; neuester Checkpoint gegen ältere")
    ap.add_argument("--a", type=Path, help="Checkpoint A (statt --run)")
    ap.add_argument("--b", type=Path, help="Checkpoint B")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--team-size", type=int, default=1)
    ap.add_argument("--opponents", type=int, default=4, help="Wie viele ältere Checkpoints")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--ratings", type=Path, default=RATINGS_PATH)
    a = ap.parse_args()

    pairs: list[tuple[Path, Path]] = []
    if a.a and a.b:
        pairs.append((a.a, a.b))
    elif a.run:
        ckpts = checkpoints_of(a.run)
        if len(ckpts) < 2:
            raise SystemExit(f"Mindestens zwei Checkpoints nötig, gefunden: {len(ckpts)}")
        latest = ckpts[-1]
        step = max(1, (len(ckpts) - 1) // a.opponents)
        for older in ckpts[:-1][::step][:a.opponents]:
            pairs.append((latest, older))
    else:
        raise SystemExit("Entweder --run oder --a/--b angeben")

    ratings = load_ratings(a.ratings)
    print(f"Standardfehler bei {a.games} Spielen und p=0,5: "
          f"+-{standard_error(0.5, a.games) * 100:.1f} Prozentpunkte "
          f"(fuer +-5 %: {games_needed(0.05)} Spiele)")

    for path_a, path_b in pairs:
        pa = path_a / "PPO_POLICY.lt" if path_a.is_dir() else path_a
        pb = path_b / "PPO_POLICY.lt" if path_b.is_dir() else path_b
        print(f"\n{pa.parent.name} vs {pb.parent.name} ({a.games} Spiele) ...")
        result = run_duel(pa, pb, a.games, a.team_size, a.deterministic)
        share = result.goal_share_a
        print(f"  Tore {result.goals_a}:{result.goals_b}   Siege {result.wins_a}:{result.wins_b} "
              f"({result.draws} remis)   Toranteil A {share * 100:.1f} % "
              f"+-{standard_error(share, max(1, result.goals_a + result.goals_b)) * 100:.1f}")
        ratings = update_ratings(ratings, result)

    save_ratings(ratings, a.ratings)
    print_table(ratings)
    print(f"\nGespeichert: {a.ratings}")


if __name__ == "__main__":
    main()
