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
- Rückweg (Review-Befund R8): Umgebungsvariable RLBOT_OBS_DELAY (0 bis 7, Default 7) setzt die
  Verzögerung in Ticks. 0 ist exakt das Verhalten vor H1: Entscheidung auf dem aktuellen Paket,
  keine Sonderbehandlung von Replay, Countdown und Pause.
      $env:RLBOT_OBS_DELAY = "0"   # vor dem Start von RLBot, in derselben Shell
- Geskripteter Anstoß (Spieltest, AUDIT.md §8.4), Standard AUS: RLBOT_SCRIPTED_KICKOFF = speedflip
  (oder frontflip, boost) steuert den Anstoß pro Tick mit deploy/scripted_kickoff.py, solange die
  Spielphase "Kickoff" ist und der Ball ruhend in der Mitte liegt, höchstens KICKOFF_SCRIPT_MAX_S.
  Danach entscheidet sofort wieder die Policy; die Aktionshistorie enthält die nächsten
  Tabelleneinträge der Skript-Eingaben. Im Teamspiel fährt nur der ballnächste Mitspieler das Skript.
      $env:RLBOT_SCRIPTED_KICKOFF = "speedflip"
- Zweiter Stand als Gegner (B6): `bot.py --policy <datei>` lädt statt policy.pt eine andere Datei
  (relativ zu diesem Ordner). bot_alt.toml startet so policy_alt.pt als "Lucy alt (RLbot)".
"""
from __future__ import annotations

import os
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
from deploy.rotation import euler_to_rotmat  # noqa: E402
from deploy.scripted_kickoff import KickoffCar, ScriptedKickoff  # noqa: E402
from env.obs_python import build_obs, obs_size  # noqa: E402

TICK_SKIP = 8
# Beobachtungslatenz des Trainings in Ticks (Gym::actionDelay = tickSkip - 1). Default des Bots;
# über RLBOT_OBS_DELAY änderbar, 0 = Verhalten vor Audit H1 (Review-Befund R8).
OBS_DELAY = TICK_SKIP - 1
OBS_DELAY_ENV = "RLBOT_OBS_DELAY"
MAX_PLAYERS = 3
ACTION_STACK = 5
DEFAULT_POLICY = Path(__file__).parent / "policy.pt"

# Phasen, in denen das Spiel kontinuierlich weiterläuft. In allen anderen (Countdown,
# Tor-Replay, Pause, Ende) springt der Zustand; Paketpuffer und Aktionshistorie werden dann
# geleert, damit die erste Entscheidung danach wie im Training nach einem Reset aussieht:
# frisches Paket, leerer Aktions-Stack.
CONTINUOUS_PHASES = frozenset({MatchPhase.Kickoff, MatchPhase.Active})

SCRIPTED_KICKOFF_ENV = "RLBOT_SCRIPTED_KICKOFF"
SCRIPTED_KICKOFF_VARIANTS = ("speedflip", "frontflip", "boost")
# Sicherheitsgrenze: länger steuert das Skript nie (Speedflip braucht höchstens ~2,5 s bis zum Ball)
KICKOFF_SCRIPT_MAX_S = 4.0
TICK_RATE = 120


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


def obs_delay_from_env(environ=None) -> int:
    """Verzögerung in Ticks aus RLBOT_OBS_DELAY (leer/fehlend = OBS_DELAY = 7).

    Erlaubt sind 0 (Verhalten vor H1) bis TICK_SKIP - 1 (wie im Training); mehr wäre älter als
    alles, was die Policy im Training gesehen hat.
    """
    raw = (os.environ if environ is None else environ).get(OBS_DELAY_ENV, "").strip()
    if raw == "":
        return OBS_DELAY
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{OBS_DELAY_ENV}={raw!r} ist keine ganze Zahl (erlaubt 0 bis {TICK_SKIP - 1})") from None
    if not 0 <= value <= TICK_SKIP - 1:
        raise ValueError(f"{OBS_DELAY_ENV}={value} außerhalb 0 bis {TICK_SKIP - 1} "
                         f"(0 = Verhalten vor Audit H1, {OBS_DELAY} = wie im Training)")
    return value


def policy_path_from_argv(argv=None) -> Path:
    """Policy-Datei aus `--policy <datei>` der Kommandozeile (B6), sonst policy.pt neben bot.py.

    Relative Pfade gelten ab dem Bot-Ordner, egal aus welchem Ordner RLBot den Bot startet. So
    können zwei Einträge (bot.toml, bot_alt.toml) denselben Code mit verschiedenen Ständen starten.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if "--policy" not in args:
        return DEFAULT_POLICY
    i = args.index("--policy")
    if i + 1 >= len(args) or not args[i + 1].strip():
        raise ValueError("--policy braucht eine Datei, z. B. --policy policy_alt.pt")
    path = Path(args[i + 1])
    return path if path.is_absolute() else DEFAULT_POLICY.parent / path


def scripted_kickoff_from_env(environ=None) -> str | None:
    """Variante aus RLBOT_SCRIPTED_KICKOFF; leer, "0", "off" oder "aus" = aus (Standard)."""
    raw = (os.environ if environ is None else environ).get(SCRIPTED_KICKOFF_ENV, "").strip().lower()
    if raw in ("", "0", "off", "aus"):
        return None
    if raw not in SCRIPTED_KICKOFF_VARIANTS:
        raise ValueError(f"{SCRIPTED_KICKOFF_ENV}={raw!r} unbekannt "
                         f"(erlaubt: {', '.join(SCRIPTED_KICKOFF_VARIANTS)}, 0 = aus)")
    return raw


def _ball_resting_at_center(packet: GamePacket) -> bool:
    if not packet.balls:
        return False
    phys = packet.balls[0].physics
    loc, vel = phys.location, phys.velocity
    return abs(loc.x) < 1 and abs(loc.y) < 1 and loc.z < 100 and abs(vel.x) + abs(vel.y) + abs(vel.z) < 1


def kickoff_car_from_packet(packet: GamePacket, index: int) -> KickoffCar:
    """Sicht des Skripts auf das eigene Auto (Richtungsvektoren wie RocketSim, deploy/rotation.py)."""
    info = packet.players[index]
    phys = info.physics
    rot = euler_to_rotmat(phys.rotation.pitch, phys.rotation.yaw, phys.rotation.roll)
    return KickoffCar(pos=np.array([phys.location.x, phys.location.y, phys.location.z]),
                      forward=rot[0], right=rot[1], up=rot[2],
                      vel=np.array([phys.velocity.x, phys.velocity.y, phys.velocity.z]),
                      on_ground=int(info.air_state) == 0)


def _nearest_entry(action) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32)
    return LOOKUP_TABLE[int(np.argmin(np.abs(LOOKUP_TABLE - a).sum(1)))]


def _closest_teammate_to_ball(packet: GamePacket, index: int) -> bool:
    """Im Teamspiel fährt nur der ballnächste Mitspieler den Anstoß (Gleichstand: kleinerer Index)."""
    me = packet.players[index]
    ball = packet.balls[0].physics.location

    def dist(p) -> float:
        return ((p.physics.location.x - ball.x) ** 2 + (p.physics.location.y - ball.y) ** 2) ** 0.5

    mine = dist(me)
    for i, other in enumerate(packet.players):
        if i != index and other.team == me.team:
            d = dist(other)
            if d < mine - 1e-3 or (abs(d - mine) <= 1e-3 and i < index):
                return False
    return True


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
        policy_path = policy_path_from_argv()
        if not policy_path.exists():
            raise FileNotFoundError(
                f"{policy_path} fehlt. Export mit:\n"
                f"  python tools/export_policy.py runs/<lauf>/checkpoints --out {policy_path}"
            )
        self.policy = load_policy(policy_path)
        check_policy_compatible(self.policy)
        self.logger.info(f"Policy geladen: {policy_path.name} ({self.policy.meta.timesteps:,} Steps), "
                         f"{self.policy.meta.layer_sizes}, "
                         f"Obs {self.policy.meta.obs_size}, Aktionen {self.policy.meta.action_count}")

        pad_locations = np.array([[p.location.x, p.location.y, p.location.z]
                                  for p in self.field_info.boost_pads])
        self.pad_index_map = build_pad_index_map(pad_locations)
        self.deterministic = True
        self._init_runtime_state()
        self.logger.info(f"Beobachtungsverzögerung {self.obs_delay} Ticks"
                         f"{' (Verhalten vor Audit H1)' if self.obs_delay == 0 else ''}")

    def _init_runtime_state(self, obs_delay: int | None = None) -> None:
        """Alles, was pro Spiel neu anfängt (auch von den Tests genutzt).

        obs_delay None = aus RLBOT_OBS_DELAY bzw. Default OBS_DELAY (7).
        """
        self.obs_delay = obs_delay_from_env() if obs_delay is None else obs_delay
        self.kickoff_variant = scripted_kickoff_from_env()
        self.kickoff_script: ScriptedKickoff | None = None
        self.kickoff_start_frame = -1
        self.kickoff_done = False       # Skript für diesen Anstoß beendet oder nicht zuständig
        self.action_history: deque[np.ndarray] = deque(maxlen=ACTION_STACK)
        self.current_action = LOOKUP_TABLE[0] * 0.0   # Nullaktion bis zur ersten Entscheidung
        self.next_decision_frame = -1
        self.packet_buffer = PacketBuffer(self.obs_delay)
        # Abstand (Ticks) zwischen dem Entscheidungs-Frame und dem benutzten Paket, zur Diagnose
        self.last_obs_delay: int | None = None

    def get_output(self, packet: GamePacket) -> ControllerState:
        if not packet.players or self.index >= len(packet.players):
            return ControllerState()

        frame = packet.match_info.frame_num
        scripted = self._scripted_kickoff(packet, frame)
        if scripted is not None:
            return scripted

        if self.obs_delay == 0:
            # Rückweg (R8): exakt das Verhalten vor Audit H1, aktuelles Paket, keine Phasen-Logik
            if frame >= self.next_decision_frame:
                self.next_decision_frame = frame + TICK_SKIP
                self.last_obs_delay = 0
                self.current_action = self._decide(packet)
            return self._to_controller(self.current_action)

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

    def _scripted_kickoff(self, packet: GamePacket, frame: int) -> ControllerState | None:
        """Controller des geskripteten Anstoßes oder None (dann entscheidet die Policy)."""
        phase = getattr(packet.match_info, "match_phase", None)
        in_kickoff = (self.kickoff_variant is not None and phase == MatchPhase.Kickoff
                      and _ball_resting_at_center(packet))
        if not in_kickoff:
            if self.kickoff_script is not None:
                self.next_decision_frame = -1          # Übergabe: Policy entscheidet sofort
            self.kickoff_script = None
            self.kickoff_done = False
            return None
        if self.kickoff_done:
            return None
        if self.kickoff_script is None:
            if not _closest_teammate_to_ball(packet, self.index):
                self.kickoff_done = True
                return None
            self.kickoff_script = ScriptedKickoff(variant=self.kickoff_variant)
            self.kickoff_start_frame = frame
        elapsed = frame - self.kickoff_start_frame
        if elapsed > KICKOFF_SCRIPT_MAX_S * TICK_RATE:
            self.kickoff_script = None
            self.kickoff_done = True
            self.next_decision_frame = -1
            return None
        self.kickoff_script.tick = elapsed              # robust gegen verpasste Pakete
        ball = packet.balls[0].physics.location
        action = self.kickoff_script.step(kickoff_car_from_packet(packet, self.index),
                                          np.array([ball.x, ball.y, ball.z]))
        if self.obs_delay > 0:
            self.packet_buffer.push(frame, packet)
        if elapsed % TICK_SKIP == 0:
            self.action_history.append(_nearest_entry(action).astype(np.float64))
        self.current_action = np.asarray(action, dtype=np.float32)
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
    # Die Agent-ID kommt beim Start über RLBot aus RLBOT_AGENT_ID (bot.toml bzw. bot_alt.toml)
    RLbotAgent("rlbot/lucy").run()
