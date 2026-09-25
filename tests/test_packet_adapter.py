"""Tests für die Übersetzung RLBot-Paket -> Obs-Sicht.

Die Pakete werden synthetisch gebaut, es läuft kein Spiel.
"""
from __future__ import annotations

import numpy as np
import pytest

flat = pytest.importorskip("rlbot_flatbuffers")

from deploy.packet_adapter import (build_pad_index_map, pad_timers_from_packet,  # noqa: E402
                                   player_from_packet, view_from_packet)
from deploy.rotation import forward_up  # noqa: E402
from env.obs_python import (BIG_PAD_COOLDOWN, BOOST_LOCATIONS, IS_BIG_PAD,  # noqa: E402
                            SMALL_PAD_COOLDOWN, build_obs, obs_size)

ON_GROUND = flat.AirState.OnGround
IN_AIR = flat.AirState.InAir
JUMPING = flat.AirState.Jumping
DOUBLE_JUMPING = flat.AirState.DoubleJumping
DODGING = flat.AirState.Dodging


def make_player(pos=(0, 0, 17), vel=(0, 0, 0), ang_vel=(0, 0, 0), rot=(0.0, 0.0, 0.0),
                team=0, boost=50.0, air_state=ON_GROUND, has_jumped=False,
                has_double_jumped=False, has_dodged=False, dodge_timeout=-1.0,
                demolished_timeout=-1.0, is_supersonic=False):
    return flat.PlayerInfo(
        physics=flat.Physics(
            location=flat.Vector3(*pos), velocity=flat.Vector3(*vel),
            angular_velocity=flat.Vector3(*ang_vel),
            rotation=flat.Rotator(pitch=rot[0], yaw=rot[1], roll=rot[2]),
        ),
        team=team, boost=boost, air_state=air_state, has_jumped=has_jumped,
        has_double_jumped=has_double_jumped, has_dodged=has_dodged,
        dodge_timeout=dodge_timeout, demolished_timeout=demolished_timeout,
        is_supersonic=is_supersonic,
    )


def make_packet(players, ball_pos=(0, 0, 93), ball_vel=(0, 0, 0), pads=None, frame=0,
                phase=None):
    """Synthetisches Paket. Standardphase ist Active (laufendes Spiel)."""
    pads = pads or [flat.BoostPadState(is_active=True, timer=0.0) for _ in BOOST_LOCATIONS]
    ball = flat.BallInfo(physics=flat.Physics(
        location=flat.Vector3(*ball_pos), velocity=flat.Vector3(*ball_vel),
        angular_velocity=flat.Vector3(0, 0, 0), rotation=flat.Rotator(0, 0, 0)))
    phase = flat.MatchPhase.Active if phase is None else phase
    return flat.GamePacket(balls=[ball], players=players, boost_pads=pads,
                           match_info=flat.MatchInfo(frame_num=frame, match_phase=phase))


def identity_pad_map():
    return build_pad_index_map(BOOST_LOCATIONS)


# --- Pad-Zuordnung -------------------------------------------------------

def test_pad_map_is_identity_for_matching_order():
    assert np.array_equal(identity_pad_map(), np.arange(len(BOOST_LOCATIONS)))


def test_pad_map_handles_shuffled_order():
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(BOOST_LOCATIONS))
    mapping = build_pad_index_map(BOOST_LOCATIONS[perm])
    # mapping[i] ist der Index im gemischten Array, dort muss das RLGym-Pad i stehen
    for rlgym_idx, packet_idx in enumerate(mapping):
        assert np.allclose(BOOST_LOCATIONS[perm][packet_idx][:2], BOOST_LOCATIONS[rlgym_idx][:2])


def test_pad_map_rejects_wrong_count():
    with pytest.raises(ValueError):
        build_pad_index_map(BOOST_LOCATIONS[:10])


def test_pad_map_rejects_unknown_positions():
    broken = BOOST_LOCATIONS.copy()
    broken[5] = [9999, 9999, 70]
    with pytest.raises(ValueError):
        build_pad_index_map(broken)


# --- Boost-Timer ---------------------------------------------------------

def test_active_pads_have_timer_zero():
    pads = [flat.BoostPadState(is_active=True, timer=0.0) for _ in BOOST_LOCATIONS]
    timers = pad_timers_from_packet(pads, identity_pad_map())
    assert np.array_equal(timers, np.zeros(len(BOOST_LOCATIONS)))


def test_timer_is_converted_to_remaining_cooldown():
    """RLBot zählt seit dem Aufsammeln hoch, RLGym zählt bis zum Respawn herunter."""
    pads = [flat.BoostPadState(is_active=False, timer=3.0) for _ in BOOST_LOCATIONS]
    timers = pad_timers_from_packet(pads, identity_pad_map())

    big = np.flatnonzero(IS_BIG_PAD)[0]
    small = np.flatnonzero(~IS_BIG_PAD)[0]
    assert timers[big] == pytest.approx(BIG_PAD_COOLDOWN - 3.0)     # 7 s Restzeit
    assert timers[small] == pytest.approx(SMALL_PAD_COOLDOWN - 3.0)  # 1 s Restzeit


def test_timer_never_goes_negative():
    pads = [flat.BoostPadState(is_active=False, timer=99.0) for _ in BOOST_LOCATIONS]
    timers = pad_timers_from_packet(pads, identity_pad_map())
    assert timers.min() >= 0.0


def test_six_big_pads_exist():
    assert IS_BIG_PAD.sum() == 6


# --- Spielerfelder -------------------------------------------------------

def test_rotation_and_scalars_are_converted():
    info = make_player(pos=(100, 200, 300), vel=(1, 2, 3), ang_vel=(0.1, 0.2, 0.3),
                       rot=(0.3, -1.2, 2.5), boost=73.0, is_supersonic=True)
    view = player_from_packet(info, car_id=0)

    fwd, up = forward_up(0.3, -1.2, 2.5)
    assert np.allclose(view.forward, fwd)
    assert np.allclose(view.up, up)
    assert np.allclose(view.pos, [100, 200, 300])
    assert view.boost == pytest.approx(0.73)   # Skala 0..100 -> 0..1
    assert view.supersonic is True


@pytest.mark.parametrize("air_state,on_ground,jumping,flipping", [
    (ON_GROUND, True, False, False),
    (JUMPING, False, True, False),
    (DOUBLE_JUMPING, False, True, False),
    (DODGING, False, False, True),
    (IN_AIR, False, False, False),
])
def test_air_state_mapping(air_state, on_ground, jumping, flipping):
    view = player_from_packet(make_player(air_state=air_state), car_id=0)
    assert (view.on_ground, view.jumping, view.flipping) == (on_ground, jumping, flipping)


def test_has_jump_and_has_flip():
    # Am Boden: beides verfügbar
    grounded = player_from_packet(make_player(air_state=ON_GROUND), car_id=0)
    assert grounded.has_jump and grounded.has_flip

    # Gesprungen, Dodge-Fenster offen
    jumped = player_from_packet(
        make_player(air_state=IN_AIR, has_jumped=True, dodge_timeout=0.9), car_id=0)
    assert not jumped.has_jump and jumped.has_flip

    # Fenster abgelaufen (-1 in der Luft nach Sprung)
    expired = player_from_packet(
        make_player(air_state=IN_AIR, has_jumped=True, dodge_timeout=-1.0), car_id=0)
    assert not expired.has_flip

    # Bereits gedodged oder doppelt gesprungen
    for kwargs in ({"has_dodged": True}, {"has_double_jumped": True}):
        used = player_from_packet(
            make_player(air_state=IN_AIR, has_jumped=True, dodge_timeout=0.5, **kwargs), car_id=0)
        assert not used.has_flip

    # Flip-Reset: in der Luft ohne Sprung
    reset = player_from_packet(make_player(air_state=IN_AIR, has_jumped=False), car_id=0)
    assert reset.has_flip


def test_demolished_flag():
    alive = player_from_packet(make_player(demolished_timeout=-1.0), car_id=0)
    dead = player_from_packet(make_player(demolished_timeout=2.5), car_id=0)
    assert not alive.demoed and dead.demoed


# --- Gesamtpaket ---------------------------------------------------------

def test_view_from_packet_builds_valid_obs():
    packet = make_packet([make_player(pos=(-2048, -2560, 17), team=0),
                          make_player(pos=(2048, 2560, 17), team=1)],
                         ball_pos=(0, 0, 93))
    view = view_from_packet(packet, identity_pad_map())
    obs = build_obs(view, view.players[0], max_players=3, action_stack_size=5)

    assert obs.shape == (obs_size(3, 5),)
    assert np.all(np.isfinite(obs))
    # Ball in der Feldmitte auf Ballradius-Höhe
    assert np.allclose(obs[:2], 0.0)
    assert obs[2] == pytest.approx(93 / 2044, abs=1e-6)
    # Eigener Block beginnt nach Ball (9) und Pad-Timern (34)
    assert obs[43] == pytest.approx(-2048 / 4096)
    assert obs[44] == pytest.approx(-2560 / 5120)


def test_both_teams_get_mirrored_obs():
    """Spiegelbildliche Aufstellung -> beide Teams sehen dasselbe."""
    packet = make_packet([make_player(pos=(-800, -2000, 17), vel=(0, 500, 0), team=0),
                          make_player(pos=(800, 2000, 17), vel=(0, -500, 0),
                                      rot=(0.0, np.pi, 0.0), team=1)])
    view = view_from_packet(packet, identity_pad_map())
    blue = build_obs(view, view.players[0])
    orange = build_obs(view, view.players[1])
    assert np.allclose(blue, orange, atol=1e-5)


def test_action_history_lands_in_obs():
    packet = make_packet([make_player(), make_player(team=1)])
    history = {0: [np.arange(8, dtype=np.float64) / 10.0]}
    view = view_from_packet(packet, identity_pad_map(), action_history=history)
    obs = build_obs(view, view.players[0])

    stack_start = 9 + 34 + 29
    assert np.allclose(obs[stack_start:stack_start + 32], 0.0)          # 4 leere Slots
    assert np.allclose(obs[stack_start + 32:stack_start + 40],
                       np.arange(8) / 10.0, atol=1e-6)                   # jüngste Aktion
