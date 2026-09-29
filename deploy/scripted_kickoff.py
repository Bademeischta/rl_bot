"""Geskripteter Anstoß als Referenz und Deployment-Kandidat (Spieltest-Auftrag, Punkt 1).

Steuert pro Tick (120 Hz), nicht über die Aktionstabelle: Gas + Boost, auf den Ball lenken und je
nach Variante
  "boost"     nur fahren,
  "frontflip" kurz vor dem Ball ein Frontflip in den Ball,
  "speedflip" früher diagonaler Flip mit sofortigem Flip-Cancel (Nick nach hinten) und
              Gegenrollen, danach Boost; zum Schluss ein Frontflip in den Ball.
Die Zeitparameter sind in RocketSim abgestimmt (eval/kickoff_eval.py --tune), je Anstoßart
(diagonal, versetzt, Mitte) eigene Werte.

Eingang ist nur, was auch RLBot liefert (Position, Richtungsvektoren, Geschwindigkeit, Bodenkontakt),
Ausgang ein 8er-Tupel wie die Aktionstabelle: throttle, steer, pitch, yaw, roll, jump, boost,
handbrake. Im RLBot-Bot ist das NICHT eingebaut (Entscheidung des Nutzers); gemessen wird es mit
eval/kickoff_eval.py gegen die gelernte Policy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TICK_RATE = 120

# Abgestimmte Parameter je Anstoßart (eval/kickoff_eval.py --tune, RocketSim, Ball in der Mitte).
#  steer_off: Kursversatz (rad) weg vom Flip-Seite vor dem Sprung; jump_tick: Sprungbeginn;
#  jump_ticks: Dauer des ersten Sprungs; gap: Ticks ohne Sprung vor dem Dodge; cancel: Ticks mit
#  Nick nach hinten nach dem Dodge; final_dist: Abstand zum Ball für den Frontflip in den Ball.
SPEEDFLIP_PARAMS = {
    # 28.09.2026, 3000 Stichproben je Art: diagonal 1,892 s (nur Boost 2,133), versetzt 2,142 s
    # (2,617), Mitte 2,467 s (2,625) bis zur ersten Berührung, Tempo dabei 2300 uu/s
    "diagonal": dict(steer_off=0.265, jump_tick=60, jump_ticks=5, gap=1, cancel=60, final_dist=700.0),
    "offset": dict(steer_off=0.154, jump_tick=54, jump_ticks=1, gap=5, cancel=77, final_dist=500.0),
    "center": dict(steer_off=0.248, jump_tick=52, jump_ticks=2, gap=5, cancel=80, final_dist=500.0),
}
FRONTFLIP_DIST = {"diagonal": 1250.0, "offset": 1250.0, "center": 1250.0}


@dataclass
class KickoffCar:
    pos: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    vel: np.ndarray
    on_ground: bool


def kickoff_kind(car_pos: np.ndarray) -> str:
    """Anstoßart aus der Startposition (Ball in der Mitte): diagonal, versetzt oder Mitte."""
    ax = abs(float(car_pos[0]))
    if ax > 1000:
        return "diagonal"
    if ax > 100:
        return "offset"
    return "center"


def _yaw_error(car: KickoffCar, target: np.ndarray) -> float:
    to = target - car.pos
    a = math.atan2(to[1], to[0]) - math.atan2(car.forward[1], car.forward[0])
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class ScriptedKickoff:
    """Ein Anstoß. `step` einmal pro Tick aufrufen, bis die Policy übernimmt (nach der ersten Berührung)."""
    variant: str = "speedflip"
    kind: str | None = None
    params: dict = field(default_factory=dict)
    tick: int = 0
    _flip_start: int | None = None
    _side: float = 0.0

    def step(self, car: KickoffCar, ball_pos: np.ndarray) -> tuple:
        if self.kind is None:
            self.kind = kickoff_kind(car.pos)
            if self.variant == "speedflip":
                self.params = {**SPEEDFLIP_PARAMS[self.kind], **self.params}
            # Flip-Seite: zur Seite, auf der der Ball liegt (bei Mitte nach rechts)
            err = _yaw_error(car, ball_pos)
            self._side = 1.0 if err >= 0 else -1.0
        t = self.tick
        self.tick += 1
        err = _yaw_error(car, ball_pos)
        dist = float(np.linalg.norm(ball_pos - car.pos))
        throttle, steer, pitch, yaw, roll, jump, boost, handbrake = 1.0, 0.0, 0.0, 0.0, 0.0, False, True, False

        if self.variant == "speedflip":
            p = self.params
            j0 = p["jump_tick"]
            dodge = j0 + p["jump_ticks"] + p["gap"]
            if t < j0:
                steer = float(np.clip(3.0 * (err - self._side * p["steer_off"]), -1, 1))
            elif t < j0 + p["jump_ticks"]:
                jump = True
            elif t < dodge:
                pass
            elif t < dodge + 2:
                jump, pitch, yaw = True, -1.0, self._side          # diagonaler Dodge Richtung Ball
            elif t < dodge + 2 + p["cancel"]:
                pitch, roll = 1.0, -self._side                     # Flip-Cancel und Gegenrollen
            elif not car.on_ground:
                pitch, roll, yaw = _level(car, err)
            else:
                steer = float(np.clip(3.0 * err, -1, 1))
                if p.get("final_dist", 0) > 0 and dist < p["final_dist"] and self._flip_start is None:
                    self._flip_start = t
            if self._flip_start is not None:
                jump, pitch = _frontflip(t - self._flip_start)
            return (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)

        steer = float(np.clip(3.0 * err, -1, 1))
        if self.variant == "frontflip":
            if self._flip_start is None and car.on_ground and dist < FRONTFLIP_DIST[self.kind or "diagonal"]:
                self._flip_start = t
            if self._flip_start is not None:
                jump, pitch = _frontflip(t - self._flip_start)
        elif self.variant != "boost":
            raise ValueError(f"Unbekannte Variante {self.variant!r}")
        return (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)


def _frontflip(dt: int) -> tuple[bool, float]:
    """Sprung 6 Ticks, 3 Ticks Pause, dann Dodge nach vorn."""
    return (dt < 6 or 9 <= dt < 12), (-1.0 if dt >= 9 else 0.0)


def _level(car: KickoffCar, yaw_err: float) -> tuple[float, float, float]:
    """In der Luft: Räder nach unten, Nase waagerecht, Richtung Ball (P-Regler)."""
    roll = float(np.clip(4.0 * car.right[2], -1, 1))
    pitch = float(np.clip(-4.0 * car.forward[2], -1, 1))
    yaw = float(np.clip(2.0 * yaw_err, -1, 1))
    return pitch, roll, yaw
