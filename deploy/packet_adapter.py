"""Übersetzt ein RLBot-v5-GamePacket in die Sicht, die der Obs-Builder erwartet.

Die heiklen Stellen sind hier gebündelt und in tests/test_packet_adapter.py geprüft:
- RLBot liefert Euler-Winkel, das Obs braucht forward/up (deploy/rotation.py)
- RLBots Boost-Timer zählt die Sekunden SEIT dem Aufsammeln, RocketSim die Restzeit BIS
  zum Respawn. Das muss umgerechnet werden, sonst sieht der Bot die Pads invertiert.
- Die Pad-Reihenfolge von RLBot ist nicht die von RLGym; sie wird über die Positionen zugeordnet.
"""
from __future__ import annotations

import numpy as np

from deploy.rotation import forward_up
from env.obs_python import (BIG_PAD_COOLDOWN, BOOST_LOCATIONS, IS_BIG_PAD, SMALL_PAD_COOLDOWN,
                            BallView, GameView, PlayerView)

# rlbot_flatbuffers.AirState
AIR_STATE_ON_GROUND = 0
AIR_STATE_JUMPING = 1
AIR_STATE_DOUBLE_JUMPING = 2
AIR_STATE_DODGING = 3


def build_pad_index_map(field_pad_locations: np.ndarray) -> np.ndarray:
    """Ordnet RLBot-Pad-Indizes den RLGym-Indizes zu (über die nächstgelegene Position).

    Rückgabe: Array der Länge 34; Position i enthält den RLBot-Index des RLGym-Pads i.
    """
    field = np.asarray(field_pad_locations, dtype=np.float64)
    if field.shape[0] != len(BOOST_LOCATIONS):
        raise ValueError(f"Erwartet {len(BOOST_LOCATIONS)} Pads, bekam {field.shape[0]}")

    mapping = np.full(len(BOOST_LOCATIONS), -1, dtype=int)
    for rlgym_idx, location in enumerate(BOOST_LOCATIONS):
        # Nur x/y vergleichen: die z-Angaben unterscheiden sich je nach Quelle leicht
        distances = np.linalg.norm(field[:, :2] - location[:2], axis=1)
        best = int(np.argmin(distances))
        if distances[best] > 100.0:
            raise ValueError(f"Kein passendes Pad für RLGym-Index {rlgym_idx} bei {location}")
        mapping[rlgym_idx] = best

    if len(set(mapping.tolist())) != len(mapping):
        raise ValueError("Pad-Zuordnung ist nicht eindeutig")
    return mapping


def pad_timers_from_packet(packet_pads, pad_index_map: np.ndarray) -> np.ndarray:
    """Restliche Abklingzeit je Pad in RLGym-Reihenfolge.

    RLBot: `timer` = Sekunden seit dem Aufsammeln, 0 wenn das Pad aktiv ist.
    RocketSim/RLGym: Sekunden bis zum Respawn. Also Restzeit = Abklingzeit - timer.
    """
    cooldowns = np.where(IS_BIG_PAD, BIG_PAD_COOLDOWN, SMALL_PAD_COOLDOWN)
    timers = np.zeros(len(BOOST_LOCATIONS), dtype=np.float64)
    for rlgym_idx, packet_idx in enumerate(pad_index_map):
        pad = packet_pads[packet_idx]
        if getattr(pad, "is_active", True):
            timers[rlgym_idx] = 0.0
        else:
            timers[rlgym_idx] = float(np.clip(cooldowns[rlgym_idx] - pad.timer,
                                              0.0, cooldowns[rlgym_idx]))
    return timers


def _vec(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=np.float64)


def _has_flip(info, air_state: int) -> bool:
    """Ist noch ein Flip/Doppelsprung verfügbar?

    RLGym rechnet: !hasDoubleJumped && !hasFlipped && airTimeSinceJump < 1.25 s.
    RLBot liefert `dodge_timeout` = Restzeit des Dodge-Fensters, aber -1 sowohl am Boden
    als auch wenn das Fenster abgelaufen ist. Die beiden Fälle müssen getrennt werden,
    sonst hätte ein stehendes Auto laut Obs keinen Flip mehr.
    """
    if info.has_double_jumped or info.has_dodged:
        return False
    if air_state == AIR_STATE_ON_GROUND:
        return True
    if not info.has_jumped:
        return True          # ohne Sprung in der Luft (heruntergefahren oder Flip-Reset)
    return info.dodge_timeout > 0


def player_from_packet(info, car_id: int) -> PlayerView:
    phys = info.physics
    fwd, up = forward_up(phys.rotation.pitch, phys.rotation.yaw, phys.rotation.roll)
    air_state = int(info.air_state)

    has_jump = not info.has_jumped
    has_flip = _has_flip(info, air_state)

    return PlayerView(
        car_id=car_id,
        team=int(info.team),
        pos=_vec(phys.location),
        forward=fwd,
        up=up,
        vel=_vec(phys.velocity),
        ang_vel=_vec(phys.angular_velocity),
        boost=float(info.boost) / 100.0,
        on_ground=air_state == AIR_STATE_ON_GROUND,
        has_flip=has_flip,
        has_jump=has_jump,
        demoed=info.demolished_timeout > 0,
        supersonic=bool(info.is_supersonic),
        flipping=air_state == AIR_STATE_DODGING,
        jumping=air_state in (AIR_STATE_JUMPING, AIR_STATE_DOUBLE_JUMPING),
    )


def view_from_packet(packet, pad_index_map: np.ndarray,
                     action_history: dict[int, list[np.ndarray]] | None = None) -> GameView:
    if not packet.balls:
        raise ValueError("Paket enthält keinen Ball")
    ball_phys = packet.balls[0].physics

    players = [player_from_packet(info, car_id=idx) for idx, info in enumerate(packet.players)]

    return GameView(
        ball=BallView(pos=_vec(ball_phys.location), vel=_vec(ball_phys.velocity),
                      ang_vel=_vec(ball_phys.angular_velocity)),
        players=players,
        pad_timers=pad_timers_from_packet(packet.boost_pads, pad_index_map),
        action_history=action_history or {},
    )
