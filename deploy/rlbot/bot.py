"""RLBot-v5-Bot, der eine im C++-Training erzeugte Policy spielt.

Start über den RLBot-Launcher mit bot.toml. Rocket League muss dafür offline und ohne
Easy Anti-Cheat laufen ("Launch without EAC"). Nie online oder ranked.

Wichtig für die Übereinstimmung mit dem Training:
- Der Bot entscheidet nur alle TICK_SKIP Ticks neu und hält die Aktion dazwischen,
  genau wie das Training mit tick_skip=8 (15 Entscheidungen pro Sekunde).
- Beobachtung, Aktionstabelle und Inferenz sind bitgleich zum Training (Paritätstests).
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

# Damit der Bot auch startet, wenn RLBot ihn aus seinem eigenen Ordner lädt
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlbot.managers import Bot  # noqa: E402
from rlbot_flatbuffers import ControllerState, GamePacket  # noqa: E402

from deploy.action_table import LOOKUP_TABLE  # noqa: E402
from deploy.packet_adapter import build_pad_index_map, view_from_packet  # noqa: E402
from deploy.policy import load_policy  # noqa: E402
from env.obs_python import build_obs  # noqa: E402

TICK_SKIP = 8
MAX_PLAYERS = 3
ACTION_STACK = 5
DEFAULT_POLICY = Path(__file__).parent / "policy.pt"


class RLbotAgent(Bot):
    def initialize(self):
        policy_path = Path(__file__).parent / "policy.pt"
        if not policy_path.exists():
            raise FileNotFoundError(
                f"{policy_path} fehlt. Export mit:\n"
                f"  python tools/export_policy.py runs/<lauf>/checkpoints --out {policy_path}"
            )
        self.policy = load_policy(policy_path)
        expected = build_obs.__module__  # nur zur Klarheit in Fehlermeldungen
        self.logger.info(f"Policy geladen: {self.policy.meta.layer_sizes}, "
                         f"Obs {self.policy.meta.obs_size}, Aktionen {self.policy.meta.action_count}")

        pad_locations = np.array([[p.location.x, p.location.y, p.location.z]
                                  for p in self.field_info.boost_pads])
        self.pad_index_map = build_pad_index_map(pad_locations)

        self.action_history: deque[np.ndarray] = deque(maxlen=ACTION_STACK)
        self.current_action = LOOKUP_TABLE[0] * 0.0   # Nullaktion bis zur ersten Entscheidung
        self.next_decision_frame = -1
        self.deterministic = True
        del expected

    def get_output(self, packet: GamePacket) -> ControllerState:
        if not packet.players or self.index >= len(packet.players):
            return ControllerState()

        frame = packet.match_info.frame_num
        if frame >= self.next_decision_frame:
            self.next_decision_frame = frame + TICK_SKIP
            self.current_action = self._decide(packet)

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
