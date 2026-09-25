"""TrueSkill-Ladder für Checkpoints (Bauplan v2 §8).

    python eval/ladder.py --run runs/lucy_1v1 --games 50
    python eval/ladder.py --a runs/x/checkpoints/100 --b runs/x/checkpoints/200 --games 100
    python eval/ladder.py --show runs/sanity/ratings.json            # nur anzeigen, auch alte Dateien
    python eval/ladder.py --migrate runs/sanity/ratings.json --out runs/sanity/ratings_v2.json

Rating-Schlüssel sind seit Audit M4 "<lauf>/<steps>". Dateien von davor haben nur "<steps>";
sie bleiben lesbar (Schlüssel werden beim Laden mit dem Laufnamen ergänzt), werden aber nie in
place umgeschrieben: --migrate schreibt eine Kopie (Review-Befund R9).

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


def rating_key(path: Path) -> str:
    """Schlüssel eines Checkpoints in ratings.json (Audit M4).

    `runs/<lauf>/checkpoints/<steps>[/PPO_POLICY.lt]` wird zu `<lauf>/<steps>`, damit gleiche
    Step-Stände aus verschiedenen Läufen (oder Netzgrößen) nicht denselben Eintrag teilen.
    Liegt der Checkpoint nicht in dieser Struktur, bleibt es beim Ordnernamen.
    """
    path = Path(path)
    ckpt = path.parent if path.name.endswith(".lt") else path
    if ckpt.parent.name == "checkpoints" and ckpt.parent.parent.name:
        return f"{ckpt.parent.parent.name}/{ckpt.name}"
    return ckpt.name


def default_ratings_path(run_dir: Path | None) -> Path:
    """Ratings liegen pro Lauf (`runs/<lauf>/ratings.json`), nicht mehr global in eval/."""
    return (run_dir / "ratings.json") if run_dir else RATINGS_PATH


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
        name_a=rating_key(path_a),
        name_b=rating_key(path_b),
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


def is_legacy_key(key: str) -> bool:
    """Schlüssel von vor Audit M4: nur die Step-Zahl ("2704829056"), ohne Laufnamen."""
    return key.isdigit()


def migrate_keys(data: dict, run_name: str) -> dict:
    """Alte Schlüssel "<steps>" -> "<lauf>/<steps>"; neue Schlüssel bleiben (Review-Befund R9)."""
    out: dict = {}
    for key, value in data.items():
        new = f"{run_name}/{key}" if is_legacy_key(key) else key
        if new in out:
            raise ValueError(f"Schlüssel {new} kommt alt und neu vor; Datei von Hand prüfen")
        out[new] = value
    return out


def has_legacy_keys(path: Path) -> bool:
    if not path.exists():
        return False
    return any(is_legacy_key(k) for k in json.loads(path.read_text(encoding="utf-8")))


def load_ratings(path: Path = RATINGS_PATH, run_name: str | None = None) -> dict[str, trueskill.Rating]:
    """Liest ratings.json. Alte Schlüssel (nur Step-Zahl) bekommen mit run_name den Laufnamen
    vorangestellt, ohne run_name bleiben sie wie sie sind. Die Datei wird dabei nie verändert."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if run_name:
        data = migrate_keys(data, run_name)
    return {k: ENV.create_rating(v["mu"], v["sigma"]) for k, v in data.items()}


def migrate_file(src: Path, dst: Path, run_name: str) -> int:
    """Schreibt eine Kopie von src mit neuen Schlüsseln nach dst. src bleibt unverändert, dst
    darf noch nicht existieren. Liefert die Zahl der umbenannten Schlüssel."""
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("Migration nur in eine Kopie, nicht in place")
    if dst.exists():
        raise FileExistsError(f"{dst} existiert schon (wird nie überschrieben)")
    data = json.loads(src.read_text(encoding="utf-8"))
    renamed = sum(1 for k in data if is_legacy_key(k))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(migrate_keys(data, run_name), indent=2), encoding="utf-8")
    return renamed


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
    ap.add_argument("--ratings", type=Path, default=None,
                    help="Standard: <run>/ratings.json bei --run, sonst eval/ratings.json")
    ap.add_argument("--exe", type=Path, default=DUEL_EXE, help="Pfad zu duel.exe")
    ap.add_argument("--run-name", default=None,
                    help="Laufname für alte Schlüssel (nur Step-Zahl); Standard: Ordnername von --run "
                         "bzw. der Ordner, in dem die ratings.json liegt")
    ap.add_argument("--show", type=Path, default=None, help="ratings.json nur anzeigen (auch alte)")
    ap.add_argument("--migrate", type=Path, default=None,
                    help="alte ratings.json in eine Kopie mit neuen Schlüsseln schreiben (--out)")
    ap.add_argument("--out", type=Path, default=None, help="Ziel für --migrate (darf nicht existieren)")
    a = ap.parse_args()

    if a.show or a.migrate:
        src = a.show or a.migrate
        run_name = a.run_name or src.resolve().parent.name
        if a.migrate:
            if not a.out:
                raise SystemExit("--migrate braucht --out <kopie>")
            n = migrate_file(a.migrate, a.out, run_name)
            print(f"{n} alte Schlüssel als '{run_name}/<steps>' nach {a.out} geschrieben; {a.migrate} unverändert")
        print_table(load_ratings(a.out if a.migrate else src, run_name))
        return

    ratings_path = a.ratings or default_ratings_path(a.run)
    if has_legacy_keys(ratings_path):
        # Nie in place umschreiben (Review-Befund R9): erst eine migrierte Kopie anlegen
        raise SystemExit(
            f"{ratings_path} enthält alte Schlüssel (nur Step-Zahl, vor Audit M4) und wird nicht "
            f"verändert.\nMigrieren in eine Kopie: python eval/ladder.py --migrate {ratings_path} "
            f"--out <kopie.json>\nund dann die Ladder mit --ratings <kopie.json> laufen lassen.")

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

    ratings = load_ratings(ratings_path, a.run_name or (a.run.name if a.run else None))
    print(f"Standardfehler bei {a.games} Spielen und p=0,5: "
          f"+-{standard_error(0.5, a.games) * 100:.1f} Prozentpunkte "
          f"(fuer +-5 %: {games_needed(0.05)} Spiele)")

    for path_a, path_b in pairs:
        pa = path_a / "PPO_POLICY.lt" if path_a.is_dir() else path_a
        pb = path_b / "PPO_POLICY.lt" if path_b.is_dir() else path_b
        print(f"\n{rating_key(pa)} vs {rating_key(pb)} ({a.games} Spiele) ...")
        result = run_duel(pa, pb, a.games, a.team_size, a.deterministic, exe=a.exe)
        share = result.goal_share_a
        print(f"  Tore {result.goals_a}:{result.goals_b}   Siege {result.wins_a}:{result.wins_b} "
              f"({result.draws} remis)   Toranteil A {share * 100:.1f} % "
              f"+-{standard_error(share, max(1, result.goals_a + result.goals_b)) * 100:.1f}")
        ratings = update_ratings(ratings, result)

    save_ratings(ratings, ratings_path)
    print_table(ratings)
    print(f"\nGespeichert: {ratings_path}")


if __name__ == "__main__":
    main()
