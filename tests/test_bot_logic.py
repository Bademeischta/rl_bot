"""Tests der Bot-Logik ohne laufendes Spiel: Tick-Skip, Aktionshistorie, Controller-Mapping."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rlbot_flatbuffers")
pytest.importorskip("rlbot")

from deploy.action_table import LOOKUP_TABLE  # noqa: E402
from deploy.packet_adapter import build_pad_index_map  # noqa: E402
from deploy.policy import Policy, PolicyMeta, _build_sequential  # noqa: E402
from deploy.rlbot.bot import ACTION_STACK, TICK_SKIP, RLbotAgent  # noqa: E402
from env.obs_python import BOOST_LOCATIONS, obs_size  # noqa: E402
from tests.test_packet_adapter import make_packet, make_player  # noqa: E402

from collections import deque  # noqa: E402


def make_agent(action_index: int = 7) -> RLbotAgent:
    """Agent ohne RLBot-Verbindung, mit einer Policy, die immer dieselbe Aktion wählt."""
    agent = object.__new__(RLbotAgent)

    seq = _build_sequential(obs_size(3, 5), len(LOOKUP_TABLE), [8])
    meta = PolicyMeta(obs_size=obs_size(3, 5), action_count=len(LOOKUP_TABLE), layer_sizes=[8])
    policy = Policy(seq, meta)
    policy.act = lambda obs, deterministic=True: action_index  # type: ignore[assignment]

    agent.policy = policy
    agent.pad_index_map = build_pad_index_map(BOOST_LOCATIONS)
    agent.action_history = deque(maxlen=ACTION_STACK)
    agent.current_action = LOOKUP_TABLE[0] * 0.0
    agent.next_decision_frame = -1
    agent.deterministic = True
    agent.index = 0
    return agent


def test_controller_mapping_is_exact():
    action = np.array([0.5, -0.25, 1.0, -1.0, 0.75, 1.0, 1.0, 0.0])
    ctrl = RLbotAgent._to_controller(action)
    assert ctrl.throttle == pytest.approx(0.5)
    assert ctrl.steer == pytest.approx(-0.25)
    assert ctrl.pitch == pytest.approx(1.0)
    assert ctrl.yaw == pytest.approx(-1.0)
    assert ctrl.roll == pytest.approx(0.75)
    assert ctrl.jump is True
    assert ctrl.boost is True
    assert ctrl.handbrake is False


def test_every_table_entry_maps_to_a_valid_controller():
    for action in LOOKUP_TABLE:
        ctrl = RLbotAgent._to_controller(action)
        for value in (ctrl.throttle, ctrl.steer, ctrl.pitch, ctrl.yaw, ctrl.roll):
            assert -1.0 <= value <= 1.0
        assert isinstance(ctrl.jump, bool)


def test_decision_is_held_for_tick_skip_frames():
    agent = make_agent(action_index=7)
    players = [make_player(), make_player(team=1)]

    first = agent.get_output(make_packet(players, frame=0))
    assert len(agent.action_history) == 1

    # Innerhalb des Tick-Skip-Fensters darf nicht neu entschieden werden
    for frame in range(1, TICK_SKIP):
        held = agent.get_output(make_packet(players, frame=frame))
        assert (held.throttle, held.steer, held.jump) == (first.throttle, first.steer, first.jump)
        assert len(agent.action_history) == 1

    agent.get_output(make_packet(players, frame=TICK_SKIP))
    assert len(agent.action_history) == 2


def test_action_history_is_capped_at_stack_size():
    agent = make_agent()
    players = [make_player(), make_player(team=1)]
    for i in range(20):
        agent.get_output(make_packet(players, frame=i * TICK_SKIP))
    assert len(agent.action_history) == ACTION_STACK


def test_chosen_action_matches_lookup_table():
    for index in (0, 13, 44, 89):
        agent = make_agent(action_index=index)
        ctrl = agent.get_output(make_packet([make_player(), make_player(team=1)], frame=0))
        expected = LOOKUP_TABLE[index]
        assert ctrl.throttle == pytest.approx(expected[0])
        assert ctrl.jump == bool(expected[5] >= 0.5)
        assert ctrl.boost == bool(expected[6] >= 0.5)


def test_missing_player_returns_neutral_controller():
    agent = make_agent()
    agent.index = 5   # Index außerhalb des Pakets
    ctrl = agent.get_output(make_packet([make_player()], frame=0))
    assert ctrl.throttle == 0.0 and not ctrl.jump


def test_obs_fed_to_policy_has_training_size():
    agent = make_agent()
    seen = {}

    def capture(obs, deterministic=True):
        seen["shape"] = obs.shape
        return 3

    agent.policy.act = capture  # type: ignore[assignment]
    agent.get_output(make_packet([make_player(), make_player(team=1)], frame=0))
    assert seen["shape"] == (obs_size(3, 5),)
