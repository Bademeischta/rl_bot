"""Python-Nachbau des C++-Obs-Builders (env/cpp/Obs.h).

Wird beim Deployment über RLBot gebraucht, wo die Beobachtung in Python gebaut wird.
Das Layout muss exakt dem Training entsprechen, sonst sieht der Bot im Spiel etwas anderes
als im Training. tests/test_obs_parity.py prüft das gegen Golden-Fixtures aus dem C++-Code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Muss zu RLGymSim_CPP/Utils/CommonValues.h passen
SIDE_WALL_X = 4096.0
BACK_WALL_Y = 5120.0
CEILING_Z = 2044.0
CAR_MAX_SPEED = 2300.0
CAR_MAX_ANG_VEL = 5.5
BOOST_LOCATIONS_AMOUNT = 34

POS_COEF = np.array([1 / SIDE_WALL_X, 1 / BACK_WALL_Y, 1 / CEILING_Z], dtype=np.float64)
VEL_COEF = 1 / CAR_MAX_SPEED
ANG_VEL_COEF = 1 / CAR_MAX_ANG_VEL
PAD_TIMER_COEF = 1 / 10.0

PLAYER_FEATURES = 29
BALL_FEATURES = 9
ACTION_ELEMS = 8

# Inversion für Orange: 180-Grad-Drehung um Z (RLGymSim PhysObj::Invert)
INVERT = np.array([-1.0, -1.0, 1.0])

BLUE, ORANGE = 0, 1

# Reihenfolge und Positionen wie RLGymSim_CPP CommonValues::BOOST_LOCATIONS.
# Der Obs-Vektor erwartet die Pad-Timer genau in dieser Reihenfolge.
BOOST_LOCATIONS = np.array([
    [0.0, -4240.0, 70.0], [-1792.0, -4184.0, 70.0], [1792.0, -4184.0, 70.0],
    [-3072.0, -4096.0, 73.0], [3072.0, -4096.0, 73.0], [-940.0, -3308.0, 70.0],
    [940.0, -3308.0, 70.0], [0.0, -2816.0, 70.0], [-3584.0, -2484.0, 70.0],
    [3584.0, -2484.0, 70.0], [-1788.0, -2300.0, 70.0], [1788.0, -2300.0, 70.0],
    [-2048.0, -1036.0, 70.0], [0.0, -1024.0, 70.0], [2048.0, -1036.0, 70.0],
    [-3584.0, 0.0, 73.0], [-1024.0, 0.0, 70.0], [1024.0, 0.0, 70.0],
    [3584.0, 0.0, 73.0], [-2048.0, 1036.0, 70.0], [0.0, 1024.0, 70.0],
    [2048.0, 1036.0, 70.0], [-1788.0, 2300.0, 70.0], [1788.0, 2300.0, 70.0],
    [-3584.0, 2484.0, 70.0], [3584.0, 2484.0, 70.0], [0.0, 2816.0, 70.0],
    [-940.0, 3310.0, 70.0], [940.0, 3308.0, 70.0], [-3072.0, 4096.0, 73.0],
    [3072.0, 4096.0, 73.0], [-1792.0, 4184.0, 70.0], [1792.0, 4184.0, 70.0],
    [0.0, 4240.0, 70.0],
], dtype=np.float64)

# Ein Pad mit z = 73 ist ein großer Boost (100), die übrigen sind kleine (12).
IS_BIG_PAD = BOOST_LOCATIONS[:, 2] > 71.0
BIG_PAD_COOLDOWN = 10.0
SMALL_PAD_COOLDOWN = 4.0


@dataclass
class BallView:
    pos: np.ndarray
    vel: np.ndarray
    ang_vel: np.ndarray


@dataclass
class PlayerView:
    car_id: int
    team: int
    pos: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    vel: np.ndarray
    ang_vel: np.ndarray
    boost: float           # 0..1
    on_ground: bool
    has_flip: bool
    has_jump: bool
    demoed: bool
    supersonic: bool
    flipping: bool
    jumping: bool


@dataclass
class GameView:
    ball: BallView
    players: list[PlayerView]
    pad_timers: np.ndarray  # 34 Werte in Sekunden, 0 = Pad verfügbar
    # Aktionshistorie je car_id, älteste zuerst, je 8 Werte
    action_history: dict[int, list[np.ndarray]] = field(default_factory=dict)


def obs_size(max_players: int = 3, action_stack_size: int = 5) -> int:
    return (BALL_FEATURES + BOOST_LOCATIONS_AMOUNT
            + PLAYER_FEATURES
            + action_stack_size * ACTION_ELEMS
            + (max_players - 1) * PLAYER_FEATURES
            + max_players * PLAYER_FEATURES)


def _inv(vec: np.ndarray, inverted: bool) -> np.ndarray:
    return vec * INVERT if inverted else vec


def _player_block(p: PlayerView, ball_pos: np.ndarray, ball_vel: np.ndarray,
                  inverted: bool) -> list[float]:
    pos = _inv(p.pos, inverted)
    vel = _inv(p.vel, inverted)
    out: list[float] = []
    out += list(pos * POS_COEF)
    out += list(_inv(p.forward, inverted))
    out += list(_inv(p.up, inverted))
    out += list(vel * VEL_COEF)
    out += list(_inv(p.ang_vel, inverted) * ANG_VEL_COEF)
    out += [float(p.boost), float(p.on_ground), float(p.has_flip), float(p.has_jump),
            float(p.demoed), float(p.supersonic), float(p.flipping), float(p.jumping)]
    out += list((ball_pos - pos) * POS_COEF)
    out += list((ball_vel - vel) * VEL_COEF)
    assert len(out) == PLAYER_FEATURES
    return out


def build_obs(view: GameView, player: PlayerView, max_players: int = 3,
              action_stack_size: int = 5) -> np.ndarray:
    """Baut den Beobachtungsvektor aus Sicht von `player`.

    Die Slot-Reihenfolge ist hier fest (Eingabereihenfolge der Spieler). Im Training wird
    gemischt, was die Policy gegenüber der Reihenfolge invariant machen soll; beim Deployment
    ist eine feste Reihenfolge richtig und vergleichbar.
    """
    inverted = player.team == ORANGE

    ball_pos = _inv(view.ball.pos, inverted)
    ball_vel = _inv(view.ball.vel, inverted)
    ball_ang_vel = _inv(view.ball.ang_vel, inverted)

    out: list[float] = []
    out += list(ball_pos * POS_COEF)
    out += list(ball_vel * VEL_COEF)
    out += list(ball_ang_vel * ANG_VEL_COEF)

    # RocketSim spiegelt die Pad-Reihenfolge; die Positionsliste ist paarweise symmetrisch,
    # invertiert ist sie also schlicht umgedreht.
    timers = np.asarray(view.pad_timers, dtype=np.float64)
    if inverted:
        timers = timers[::-1]
    out += list(timers * PAD_TIMER_COEF)

    out += _player_block(player, ball_pos, ball_vel, inverted)

    history = view.action_history.get(player.car_id, [])
    history = history[-action_stack_size:]
    out += [0.0] * ((action_stack_size - len(history)) * ACTION_ELEMS)
    for action in history:
        out += list(np.asarray(action, dtype=np.float64))

    teammates, opponents = [], []
    for other in view.players:
        if other.car_id == player.car_id:
            continue
        block = _player_block(other, ball_pos, ball_vel, inverted)
        (teammates if other.team == player.team else opponents).append(block)

    if len(teammates) > max_players - 1:
        raise ValueError(f"zu viele Mitspieler: {len(teammates)} > {max_players - 1}")
    if len(opponents) > max_players:
        raise ValueError(f"zu viele Gegner: {len(opponents)} > {max_players}")

    for slots, target in ((teammates, max_players - 1), (opponents, max_players)):
        while len(slots) < target:
            slots.append([0.0] * PLAYER_FEATURES)
        for block in slots:
            out += block

    result = np.asarray(out, dtype=np.float32)
    assert result.shape[0] == obs_size(max_players, action_stack_size), result.shape
    return result
