"""RLBot-v5-Bot, der eine im C++-Training erzeugte Policy spielt.

Start über den RLBot-Launcher mit bot.toml. Rocket League muss dafür offline und ohne
Easy Anti-Cheat laufen ("Launch without EAC"). Nie online oder ranked.

Wichtig für die Übereinstimmung mit dem Training:
- Der Bot entscheidet nur alle TICK_SKIP Ticks neu und hält die Aktion dazwischen,
  genau wie das Training mit tick_skip=8 (15 Entscheidungen pro Sekunde).
- Beobachtung, Aktionstabelle und Inferenz sind bitgleich zum Training (Paritätstests).
- Audit H1: Das Training baut die Beobachtung OBS_DELAY = 7 Ticks bevor die Aktion wirkt
  (RLGymSim `Gym::actionDelay = tickSkip - 1`). Der Bot entscheidet deshalb auf dem Paket von
  vor 7 Ticks, nicht auf dem aktuellen, sonst bekäme er im Spiel frischere Daten als je im
  Training und würde seine Schüsse zu weit vorhalten.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

# Damit der Bot auch startet, wenn RLBot ihn aus seinem eigenen Ordner lädt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlbot.managers import Bot  # noqa: E402
from rlbot_flatbuffers import ControllerState, GamePacket, MatchPhase  # noqa: E402

from deploy.action_table import LOOKUP_TABLE  # noqa: E402
from deploy.packet_adapter import build_pad_index_map, view_from_packet  # noqa: E402
from deploy.policy import load_policy  # noqa: E402
from env.obs_python import build_obs, obs_size  # noqa: E402

TICK_SKIP = 8
# Beobachtungslatenz des Trainings in Ticks (Gym::actionDelay = tickSkip - 1)
OBS_DELAY = TICK_SKIP - 1
MAX_PLAYERS = 3
ACTION_STACK = 5
DEFAULT_POLICY = Path(__file__).parent / "policy.pt"

# Phasen, in denen das Spiel kontinuierlich weiterläuft. In allen anderen (Countdown,
# Tor-Replay, Pause, Ende) springt der Zustand; Paketpuffer und Aktionshistorie werden dann
# geleert, damit die erste Entscheidung danach wie im Training nach einem Reset aussieht:
# frisches Paket, leerer Aktions-Stack.
CONTINUOUS_PHASES = frozenset({MatchPhase.Kickoff, MatchPhase.Active})


class PacketBuffer:
    """Hält die letzten Pakete nach Frame-Nummer und liefert das von vor `delay` Ticks.

    - Doppelt gelieferte oder eingefrorene Frames (Pause) werden nicht erneut aufgenommen,
      sonst würden sie ältere Pakete aus dem Puffer drängen.
    - Springt die Frame-Nummer zurück (neues Spiel), beginnt der Puffer neu.
    - Fehlen Ticks (der Bot hat Pakete verpasst), wird das Paket genommen, das dem
      Zielabstand am nächsten liegt; bei Gleichstand das ältere. Frischer als `delay` Ticks
      wird nur entschieden, wenn nichts Älteres da ist (Start, nach einem Reset).
    """

    def __init__(self, delay: int = OBS_DELAY):
        self.delay = delay
        self._packets: deque[tuple[int, Any]] = deque(maxlen=delay + 1)

    def __len__(self) -> int:
        return len(self._packets)

    @property
    def frames(self) -> list[int]:
        return [f for f, _ in self._packets]

    def clear(self) -> None:
        self._packets.clear()

    def push(self, frame: int, packet: Any) -> None:
        if self._packets:
            last_frame = self._packets[-1][0]
            if frame == last_frame:
                return
            if frame < last_frame:
                self._packets.clear()
        self._packets.append((frame, packet))

    def delayed(self, frame: int) -> tuple[int, Any]:
        """(Frame, Paket) für eine Entscheidung im Frame `frame`. Puffer darf nicht leer sein."""
        if not self._packets:
            raise LookupError("PacketBuffer ist leer")
        target = frame - self.delay
        best: tuple[int, Any] | None = None
        best_dist = 0
        for entry in self._packets:            # älteste zuerst -> Gleichstand geht ans ältere
            dist = abs(entry[0] - target)
            if best is None or dist < best_dist:
                best, best_dist = entry, dist
        return best


def _is_continuous(packet: GamePacket) -> bool:
    phase = getattr(packet.match_info, "match_phase", None)
    return phase is None or phase in CONTINUOUS_PHASES


def check_policy_compatible(policy, max_players: int = MAX_PLAYERS,
                            action_stack: int = ACTION_STACK) -> None:
    """Audit M7: Policy-Eingabe/-Ausgabe müssen zum Obs-Builder und zur Aktionstabelle passen.

    Sonst fällt ein Checkpoint mit anderem max_players/action_stack_size erst beim ersten
    matmul im laufenden Spiel auf.
    """
    expected_obs = obs_size(max_players, action_stack)
    if policy.meta.obs_size != expected_obs:
        raise ValueError(
            f"Policy erwartet Obs-Größe {policy.meta.obs_size}, der Obs-Builder liefert "
            f"{expected_obs} (MAX_PLAYERS={max_players}, ACTION_STACK={action_stack}). "
            f"Passen die Konstanten in deploy/rlbot/bot.py zur Trainings-Config "
            f"(env.max_players / env.action_stack_size)? Quelle: {policy.meta.source or 'policy.pt'}")
    if policy.meta.action_count != len(LOOKUP_TABLE):
        raise ValueError(
            f"Policy hat {policy.meta.action_count} Aktionen, die Aktionstabelle {len(LOOKUP_TABLE)}. "
            f"Quelle: {policy.meta.source or 'policy.pt'}")


class RLbotAgent(Bot):
    def initialize(self):
        policy_path = Path(__file__).parent / "policy.pt"
        if not policy_path.exists():
            raise FileNotFoundError(
                f"{policy_path} fehlt. Export mit:\n"
                f"  python tools/export_policy.py runs/<lauf>/checkpoints --out {policy_path}"
            )
        self.policy = load_policy(policy_path)
        check_policy_compatible(self.policy)
        self.logger.info(f"Policy geladen: {self.policy.meta.layer_sizes}, "
                         f"Obs {self.policy.meta.obs_size}, Aktionen {self.policy.meta.action_count}")

        pad_locations = np.array([[p.location.x, p.location.y, p.location.z]
                                  for p in self.field_info.boost_pads])
        self.pad_index_map = build_pad_index_map(pad_locations)
        self.deterministic = True
        self._init_runtime_state()

    def _init_runtime_state(self) -> None:
        """Alles, was pro Spiel neu anfängt (auch von den Tests genutzt)."""
        self.action_history: deque[np.ndarray] = deque(maxlen=ACTION_STACK)
        self.current_action = LOOKUP_TABLE[0] * 0.0   # Nullaktion bis zur ersten Entscheidung
        self.next_decision_frame = -1
        self.packet_buffer = PacketBuffer(OBS_DELAY)
        # Abstand (Ticks) zwischen dem Entscheidungs-Frame und dem benutzten Paket, zur Diagnose
        self.last_obs_delay: int | None = None

    def get_output(self, packet: GamePacket) -> ControllerState:
        if not packet.players or self.index >= len(packet.players):
            return ControllerState()

        frame = packet.match_info.frame_num
        if not _is_continuous(packet):
            # Replay, Countdown, Pause: Eingaben wirken nicht, der Zustand springt danach.
            # Wie ein Reset im Training: Puffer und Aktions-Stack leeren, im ersten laufenden
            # Frame sofort neu entscheiden (auf dem frischen Paket, mit leerem Stack).
            self.packet_buffer.clear()
            self.action_history.clear()
            self.next_decision_frame = -1
            self.current_action = LOOKUP_TABLE[0] * 0.0
            return ControllerState()
        self.packet_buffer.push(frame, packet)

        if frame >= self.next_decision_frame:
            self.next_decision_frame = frame + TICK_SKIP
            delayed_frame, delayed_packet = self.packet_buffer.delayed(frame)
            self.last_obs_delay = frame - delayed_frame
            self.current_action = self._decide(delayed_packet)

        return self._to_controller(self.current_action)

    def _decide(self, packet: GamePacket) -> np.ndarray:
        view = view_from_packet(
            packet, self.pad_index_map,
            action_history={self.index: list(self.action_history)},
        )
        me = view.players[self.index]
        obs = build_obs(view, me, MAX_PLAYERS, ACTION_STACK)

        action_index = self.policy.act(obs, deterministic=self.deterministic)
        action = LOOKUP_TABLE[int(action_index)]
        self.action_history.append(action.astype(np.float64))
        return action

    @staticmethod
    def _to_controller(action: np.ndarray) -> ControllerState:
        # Reihenfolge wie RLGymSim Action: throttle, steer, pitch, yaw, roll, jump, boost, handbrake
        return ControllerState(
            throttle=float(action[0]),
            steer=float(action[1]),
            pitch=float(action[2]),
            yaw=float(action[3]),
            roll=float(action[4]),
            jump=bool(action[5] >= 0.5),
            boost=bool(action[6] >= 0.5),
            handbrake=bool(action[7] >= 0.5),
        )


if __name__ == "__main__":
    RLbotAgent("rlbot/lucy").run()
