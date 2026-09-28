"""Geskripteter Anstoß im RLBot-Bot (RLBOT_SCRIPTED_KICKOFF, Spieltest AUDIT.md §8.4).

Logik-Tests mit synthetischen Paketen und ein Durchlauf in RocketSim, in dem der Bot über echte
RLBot-Pakete (aus dem Simulator-Zustand gebaut) ein Auto steuert.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rlbot_flatbuffers")
pytest.importorskip("rlbot")

import rlbot_flatbuffers as flat  # noqa: E402

from deploy.action_table import LOOKUP_TABLE  # noqa: E402
from deploy.rlbot import bot as botmod  # noqa: E402
from deploy.rlbot.bot import (KICKOFF_SCRIPT_MAX_S, TICK_RATE, TICK_SKIP,  # noqa: E402
                              scripted_kickoff_from_env)
from tests.test_bot_logic import make_agent  # noqa: E402
from tests.test_packet_adapter import make_packet, make_player  # noqa: E402

KICKOFF = flat.MatchPhase.Kickoff
ACTIVE = flat.MatchPhase.Active


def agent_with(monkeypatch, variant: str | None, index: int = 0):
    if variant is None:
        monkeypatch.delenv(botmod.SCRIPTED_KICKOFF_ENV, raising=False)
    else:
        monkeypatch.setenv(botmod.SCRIPTED_KICKOFF_ENV, variant)
    agent = make_agent()
    agent.index = index
    calls = []

    def act(obs, deterministic=True):
        calls.append(obs)
        return 7

    agent.policy.act = act  # type: ignore[assignment]
    return agent, calls


def diag_players():
    # Blau diagonal links (schaut zum Ball), Orange gespiegelt
    return [make_player(pos=(-2048, -2560, 17), rot=(0.0, np.pi / 4, 0.0)),
            make_player(pos=(2048, 2560, 17), rot=(0.0, -3 * np.pi / 4, 0.0), team=1)]


def test_env_var_default_is_off_and_rejects_unknown_values():
    assert scripted_kickoff_from_env({}) is None
    assert scripted_kickoff_from_env({"RLBOT_SCRIPTED_KICKOFF": "0"}) is None
    assert scripted_kickoff_from_env({"RLBOT_SCRIPTED_KICKOFF": "Speedflip"}) == "speedflip"
    with pytest.raises(ValueError):
        scripted_kickoff_from_env({"RLBOT_SCRIPTED_KICKOFF": "halfflip"})


def test_without_the_env_var_the_policy_plays_the_kickoff(monkeypatch):
    agent, calls = agent_with(monkeypatch, None)
    agent.get_output(make_packet(diag_players(), frame=100, phase=KICKOFF))
    assert len(calls) == 1


def test_script_drives_the_kickoff_and_hands_over_after_the_touch(monkeypatch):
    agent, calls = agent_with(monkeypatch, "speedflip")
    for f in range(20):
        c = agent.get_output(make_packet(diag_players(), frame=100 + f, phase=KICKOFF))
        assert c.throttle == 1.0 and c.boost                       # Vollgas mit Boost
    assert calls == []                                            # Policy war nicht dran
    assert len(agent.action_history) == 3                         # Frames 0, 8, 16: nächste Tabelleneinträge
    assert all(any(np.array_equal(a, e) for e in LOOKUP_TABLE) for a in agent.action_history)
    # Ball berührt -> Phase Active: im ersten Frame entscheidet die Policy (mit gefülltem Stack)
    agent.get_output(make_packet(diag_players(), ball_vel=(500, 500, 0), frame=120, phase=ACTIVE))
    assert len(calls) == 1
    stack = calls[0][72:112].reshape(5, 8)
    assert np.abs(stack[-3:]).sum() > 0                           # Skript-Eingaben im Aktions-Stack


def test_script_gives_up_after_the_safety_limit(monkeypatch):
    agent, calls = agent_with(monkeypatch, "speedflip")
    limit = int(KICKOFF_SCRIPT_MAX_S * TICK_RATE)
    for f in range(limit + 1):
        agent.get_output(make_packet(diag_players(), frame=f, phase=KICKOFF))
    assert calls == []
    agent.get_output(make_packet(diag_players(), frame=limit + 1, phase=KICKOFF))
    assert len(calls) == 1                                        # Ball ruht noch, Policy übernimmt trotzdem


def test_in_2v2_only_the_teammate_closest_to_the_ball_runs_the_script(monkeypatch):
    players = [make_player(pos=(-2048, -2560, 17), rot=(0.0, np.pi / 4, 0.0)),       # Blau, näher
               make_player(pos=(0, -4608, 17), rot=(0.0, np.pi / 2, 0.0)),           # Blau, hinten
               make_player(pos=(2048, 2560, 17), team=1), make_player(pos=(0, 4608, 17), team=1)]
    near, near_calls = agent_with(monkeypatch, "speedflip", index=0)
    far, far_calls = agent_with(monkeypatch, "speedflip", index=1)
    for f in range(10):
        near.get_output(make_packet(players, frame=f, phase=KICKOFF))
        far.get_output(make_packet(players, frame=f, phase=KICKOFF))
    assert near_calls == [] and len(far_calls) >= 1


# --- Durchlauf in RocketSim: Bot steuert über RLBot-Pakete ------------------------------------

def _packet_from_sim(arena, cars, frame: int, phase) -> flat.GamePacket:
    players = []
    for car in cars:
        s = car.get_state()
        ang = s.rot_mat.as_angle()
        players.append(flat.PlayerInfo(
            physics=flat.Physics(location=flat.Vector3(*s.pos), velocity=flat.Vector3(*s.vel),
                                 angular_velocity=flat.Vector3(*s.ang_vel),
                                 rotation=flat.Rotator(pitch=ang.pitch, yaw=ang.yaw, roll=ang.roll)),
            team=int(car.team), boost=s.boost,
            air_state=flat.AirState.OnGround if s.is_on_ground else flat.AirState.InAir,
            has_jumped=s.has_jumped, has_double_jumped=s.has_double_jumped, has_dodged=s.has_flipped,
            dodge_timeout=-1.0, demolished_timeout=-1.0, is_supersonic=s.is_supersonic))
    b = arena.ball.get_state()
    ball = flat.BallInfo(physics=flat.Physics(location=flat.Vector3(*b.pos), velocity=flat.Vector3(*b.vel),
                                              angular_velocity=flat.Vector3(*b.ang_vel),
                                              rotation=flat.Rotator(0, 0, 0)))
    pads = [flat.BoostPadState(is_active=True, timer=0.0) for _ in range(34)]
    return flat.GamePacket(balls=[ball], players=players, boost_pads=pads,
                           match_info=flat.MatchInfo(frame_num=frame, match_phase=phase))


def test_bot_with_script_reaches_the_ball_like_the_tuned_speedflip_in_rocketsim(monkeypatch):
    pytest.importorskip("RocketSim")
    from eval.kickoff_eval import _controls, init_rocketsim, rs, spawn_seeds
    init_rocketsim()
    arena = rs.Arena(rs.GameMode.SOCCAR)
    cars = [arena.add_car(rs.Team.BLUE), arena.add_car(rs.Team.ORANGE)]
    arena.reset_kickoff(spawn_seeds(1)["diagonal_links"][0])
    t0 = arena.tick_count
    agent, calls = agent_with(monkeypatch, "speedflip")
    touch = None
    for tick in range(360):
        hit = cars[0].get_state().ball_hit_info
        touched = hit.is_valid and hit.tick_count_when_hit >= t0
        c = agent.get_output(_packet_from_sim(arena, cars, tick, ACTIVE if touched else KICKOFF))
        cars[0].set_controls(_controls([c.throttle, c.steer, c.pitch, c.yaw, c.roll,
                                        float(c.jump), float(c.boost), float(c.handbrake)]))
        arena.step(1)
        hit = cars[0].get_state().ball_hit_info
        if touch is None and hit.is_valid and hit.tick_count_when_hit >= t0:
            touch = (hit.tick_count_when_hit - t0) / 120.0
    assert touch is not None and touch < 2.0                      # abgestimmt: 1,89 s diagonal
    assert calls                                                  # danach spielt die Policy
