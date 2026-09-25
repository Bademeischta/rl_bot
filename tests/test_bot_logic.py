"""Tests der Bot-Logik ohne laufendes Spiel: Tick-Skip, Aktionshistorie, Controller-Mapping,
Paketpuffer (Audit H1)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rlbot_flatbuffers")
pytest.importorskip("rlbot")

import rlbot_flatbuffers as flat  # noqa: E402

from deploy.action_table import LOOKUP_TABLE  # noqa: E402
from deploy.packet_adapter import build_pad_index_map  # noqa: E402
from deploy.policy import Policy, PolicyMeta, _build_sequential  # noqa: E402
from deploy.rlbot.bot import ACTION_STACK, OBS_DELAY, TICK_SKIP, PacketBuffer, RLbotAgent  # noqa: E402
from env.obs_python import BOOST_LOCATIONS, SIDE_WALL_X, obs_size  # noqa: E402
from tests.test_packet_adapter import make_packet, make_player  # noqa: E402

PLAYERS = None


def two_players():
    return [make_player(), make_player(team=1)]


def make_agent(action_index: int = 7) -> RLbotAgent:
    """Agent ohne RLBot-Verbindung, mit einer Policy, die immer dieselbe Aktion wählt."""
    agent = object.__new__(RLbotAgent)

    seq = _build_sequential(obs_size(3, 5), len(LOOKUP_TABLE), [8])
    meta = PolicyMeta(obs_size=obs_size(3, 5), action_count=len(LOOKUP_TABLE), layer_sizes=[8])
    policy = Policy(seq, meta)
    policy.act = lambda obs, deterministic=True: action_index  # type: ignore[assignment]

    agent.policy = policy
    agent.pad_index_map = build_pad_index_map(BOOST_LOCATIONS)
    agent.deterministic = True
    agent.index = 0
    agent._init_runtime_state()
    return agent


def capture_ball_x(agent: RLbotAgent) -> list[float]:
    """Zeichnet die Ball-x-Position (in uu) jeder Beobachtung auf, die die Policy sieht."""
    seen: list[float] = []

    def act(obs, deterministic=True):
        seen.append(float(obs[0]) * SIDE_WALL_X)   # obs[0] = ball.pos.x * posCoef
        return 3

    agent.policy.act = act  # type: ignore[assignment]
    return seen


def feed(agent: RLbotAgent, frames, phase=None):
    """Pakete mit Ball bei x = Frame-Nummer, damit man am Obs ablesen kann, welches Paket es war."""
    for frame in frames:
        agent.get_output(make_packet(two_players(), ball_pos=(frame, 0, 93), frame=frame, phase=phase))


# --- Controller-Mapping ------------------------------------------------------

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


# --- Tick-Skip und Historie --------------------------------------------------

def test_decision_is_held_for_tick_skip_frames():
    agent = make_agent(action_index=7)
    players = two_players()

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
    players = two_players()
    for i in range(20):
        agent.get_output(make_packet(players, frame=i * TICK_SKIP))
    assert len(agent.action_history) == ACTION_STACK


def test_chosen_action_matches_lookup_table():
    for index in (0, 13, 44, 89):
        agent = make_agent(action_index=index)
        ctrl = agent.get_output(make_packet(two_players(), frame=0))
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
    agent.get_output(make_packet(two_players(), frame=0))
    assert seen["shape"] == (obs_size(3, 5),)


# --- Paketpuffer (Audit H1) --------------------------------------------------

def test_obs_delay_matches_training_after_warmup():
    """Ab der zweiten Entscheidung stammt die Beobachtung von vor OBS_DELAY = 7 Ticks."""
    assert OBS_DELAY == TICK_SKIP - 1 == 7
    agent = make_agent()
    seen = capture_ball_x(agent)
    feed(agent, range(0, 25))
    # Entscheidungen bei Frame 0, 8, 16, 24 -> Pakete 0, 1, 9, 17
    assert seen == [0.0, 1.0, 9.0, 17.0]
    assert agent.last_obs_delay == 7


def test_first_decision_uses_the_only_packet_available():
    """Beim Start (und nach jedem Reset) gibt es nichts Älteres: Delay 0, wie die
    Reset-Beobachtung im Training."""
    agent = make_agent()
    seen = capture_ball_x(agent)
    feed(agent, [0])
    assert seen == [0.0]
    assert agent.last_obs_delay == 0


def test_missed_ticks_pick_the_closest_packet_and_prefer_older_on_tie():
    """Der Bot bekommt nur jedes zweite Paket: Ziel-Frame 9 liegt zwischen 8 und 10 -> 8."""
    agent = make_agent()
    seen = capture_ball_x(agent)
    feed(agent, range(0, 17, 2))
    # Entscheidungen bei 0, 8, 16: Ziel 0 -> 0; Ziel 1 -> 0 oder 2 (Gleichstand, älter = 0);
    # Ziel 9 -> 8 oder 10 (Gleichstand, älter = 8)
    assert seen == [0.0, 0.0, 8.0]
    assert agent.last_obs_delay == 8


def test_large_gap_uses_oldest_packet_not_older_than_the_buffer():
    """Nach einer langen Lücke ist das älteste vorhandene Paket das beste."""
    agent = make_agent()
    seen = capture_ball_x(agent)
    feed(agent, [0, 1, 2, 3, 40, 41, 42, 43, 44, 45, 46, 47, 48])
    # Entscheidung bei 40: Ziel 33; Kandidaten 0..3 (Abstand >= 30) und 40 (Abstand 7) -> 40.
    # Entscheidung bei 48: Ziel 41 -> Paket 41 (Puffer hält 8 Pakete: 41..48)
    assert seen[-2:] == [40.0, 41.0]


def test_goal_replay_and_kickoff_start_fresh():
    """Tor-Replay: Pakete aus dem Replay dürfen nie die Grundlage einer Kickoff-Entscheidung
    sein, und der Aktions-Stack fängt beim Kickoff leer an (wie nach einem Reset im Training)."""
    agent = make_agent()
    seen = capture_ball_x(agent)
    feed(agent, range(0, 16))                                           # Spiel läuft
    feed(agent, range(16, 26), phase=flat.MatchPhase.GoalScored)       # Tor, Ball im Netz
    feed(agent, range(26, 36), phase=flat.MatchPhase.Replay)           # Replay: Zustand springt
    feed(agent, range(36, 41), phase=flat.MatchPhase.Countdown)        # Autos stehen am Kickoff
    replay_frames = set(range(16, 41))

    assert len(agent.action_history) == 0                              # in Countdown geleert
    assert len(agent.packet_buffer) == 0
    assert not (set(seen) & replay_frames)                             # im Replay keine Entscheidung
    seen.clear()
    feed(agent, range(41, 60), phase=flat.MatchPhase.Kickoff)          # Anpfiff
    # Sofortige Entscheidung im ersten Kickoff-Frame (Delay 0, wie die Reset-Obs im Training),
    # danach bei 49 und 57 mit Ziel 42 und 50: alles Kickoff-Pakete
    assert seen == [41.0, 42.0, 50.0]
    assert not (set(seen) & replay_frames)
    assert len(agent.action_history) == 3                              # nur Kickoff-Entscheidungen
    assert agent.last_obs_delay == 7


def test_pause_freezes_frames_without_polluting_the_buffer():
    """Während einer Pause wiederholt sich die Frame-Nummer. Danach darf die Entscheidung
    nicht auf dem eingefrorenen Paket beruhen, und der Delay bleibt <= 7 Ticks."""
    agent = make_agent()
    seen = capture_ball_x(agent)
    feed(agent, range(0, 11))
    for _ in range(30):                                                 # Pause: Frame 10 friert ein
        ctrl = agent.get_output(make_packet(two_players(), ball_pos=(10, 0, 93), frame=10,
                                            phase=flat.MatchPhase.Paused))
        assert ctrl.throttle == 0.0 and not ctrl.boost                  # neutral, keine Entscheidung
    assert len(agent.packet_buffer) == 0
    assert len(agent.action_history) == 0
    seen.clear()
    feed(agent, range(11, 25))
    # Erster laufender Frame 11: sofortige Entscheidung auf Paket 11 (Delay 0), dann bei 19
    # mit Ziel 12. Das eingefrorene Paket 10 ist nie Grundlage einer Entscheidung.
    assert seen == [11.0, 12.0]
    assert agent.last_obs_delay == 7


def test_duplicate_frames_in_active_play_are_ignored():
    """Doppelt geliefertes Paket (gleiche Frame-Nummer, Spiel läuft) verdrängt nichts."""
    buf = PacketBuffer(7)
    for frame in range(0, 8):
        buf.push(frame, f"p{frame}")
    for _ in range(20):
        buf.push(7, "dup")
    assert buf.frames == list(range(0, 8))
    assert buf.delayed(8) == (1, "p1")


def test_frame_jump_backwards_restarts_the_buffer():
    buf = PacketBuffer(7)
    for frame in range(100, 108):
        buf.push(frame, None)
    buf.push(3, "neu")                                                   # neues Spiel
    assert buf.frames == [3]
    assert buf.delayed(3) == (3, "neu")


def test_empty_buffer_raises():
    with pytest.raises(LookupError):
        PacketBuffer(7).delayed(0)


# --- Obs-Größenprüfung (Audit M7) -------------------------------------------

def make_policy(obs: int, actions: int):
    seq = _build_sequential(obs, actions, [8])
    return Policy(seq, PolicyMeta(obs_size=obs, action_count=actions, layer_sizes=[8], source="x.pt"))


def test_matching_policy_passes_check():
    from deploy.rlbot.bot import check_policy_compatible
    check_policy_compatible(make_policy(obs_size(3, 5), len(LOOKUP_TABLE)))


def test_wrong_obs_size_is_rejected_with_clear_message():
    from deploy.rlbot.bot import check_policy_compatible
    wrong = make_policy(obs_size(3, 3), len(LOOKUP_TABLE))   # action_stack_size 3 statt 5
    with pytest.raises(ValueError) as err:
        check_policy_compatible(wrong)
    msg = str(err.value)
    assert str(obs_size(3, 3)) in msg and "257" in msg
    assert "action_stack_size" in msg and "x.pt" in msg


def test_wrong_action_count_is_rejected():
    from deploy.rlbot.bot import check_policy_compatible
    with pytest.raises(ValueError, match="Aktionen"):
        check_policy_compatible(make_policy(obs_size(3, 5), 45))
