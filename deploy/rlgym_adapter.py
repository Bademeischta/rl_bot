"""Übersetzt einen RLGym-v2-GameState in die Sicht, die der Obs-Builder erwartet.

Wird für deploy/watch.py gebraucht (Selbstspiel im Python-Simulator mit RLViser-Anzeige).
Die Feldbedeutungen stimmen hier eins zu eins mit der C++-Seite überein, weil beide auf
RocketSim aufsetzen:
- `boost_pad_timers` ist in beiden die Restzeit bis zum Respawn (anders als bei RLBot).
- `has_flip` wird wie in RLGymSim gerechnet: kein Doppelsprung, kein Flip, und seit dem
  Absprung weniger als 1,25 s in der Luft.
"""
from __future__ import annotations

import numpy as np

from env.obs_python import BallView, GameView, PlayerView

# RLConst::DOUBLEJUMP_MAX_DELAY
DOUBLEJUMP_MAX_DELAY = 1.25


def _has_flip(car) -> bool:
    return (not car.has_double_jumped and not car.has_flipped
            and car.air_time_since_jump < DOUBLEJUMP_MAX_DELAY)


def player_from_car(agent_id, car, car_id: int) -> PlayerView:
    phys = car.physics
    return PlayerView(
        car_id=car_id,
        team=int(car.team_num),
        pos=np.asarray(phys.position, dtype=np.float64),
        forward=np.asarray(phys.forward, dtype=np.float64),
        up=np.asarray(phys.up, dtype=np.float64),
        vel=np.asarray(phys.linear_velocity, dtype=np.float64),
        ang_vel=np.asarray(phys.angular_velocity, dtype=np.float64),
        boost=float(car.boost_amount) / 100.0,
        on_ground=bool(car.on_ground),
        has_flip=_has_flip(car),
        has_jump=not car.has_jumped,
        demoed=car.demo_respawn_timer > 0,
        supersonic=bool(car.is_supersonic),
        flipping=bool(car.is_flipping),
        jumping=bool(car.is_jumping),
    )


def view_from_state(state, action_history: dict[int, list[np.ndarray]] | None = None
                    ) -> tuple[GameView, dict]:
    """Gibt die Sicht und die Zuordnung agent_id -> car_id zurück."""
    ids = {agent_id: idx for idx, agent_id in enumerate(state.cars)}
    players = [player_from_car(agent_id, car, ids[agent_id])
               for agent_id, car in state.cars.items()]

    view = GameView(
        ball=BallView(
            pos=np.asarray(state.ball.position, dtype=np.float64),
            vel=np.asarray(state.ball.linear_velocity, dtype=np.float64),
            ang_vel=np.asarray(state.ball.angular_velocity, dtype=np.float64),
        ),
        players=players,
        pad_timers=np.asarray(state.boost_pad_timers, dtype=np.float64),
        action_history=action_history or {},
    )
    return view, ids
