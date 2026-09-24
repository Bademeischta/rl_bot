"""Python-Nachbau der Aktions-Lookup-Tabelle aus RLGymSim_CPP (DiscreteAction.cpp).

Die Reihenfolge muss exakt stimmen: Die Policy gibt einen Index aus, und ein anderer Index
bedeutet im Spiel eine andere Eingabe. tests/test_action_table_parity.py prüft das gegen
den Dump aus dem C++-Code.

Ein Eintrag ist (throttle, steer, pitch, yaw, roll, jump, boost, handbrake).
"""
from __future__ import annotations

import numpy as np

R_B = (0.0, 1.0)
R_F = (-1.0, 0.0, 1.0)


def build_lookup_table() -> np.ndarray:
    actions: list[list[float]] = []

    # Bodenaktionen
    for throttle in R_F:
        for steer in R_F:
            for boost in R_B:
                for handbrake in R_B:
                    # Gas zusätzlich zu Boost bringt nichts
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append([throttle, steer, 0.0, steer, 0.0, 0.0, boost, handbrake])

    # Luftaktionen
    for pitch in R_F:
        for yaw in R_F:
            for roll in R_F:
                for jump in R_B:
                    for boost in R_B:
                        # Roll wird nur für den Sideflip gebraucht
                        if jump == 1 and yaw != 0:
                            continue
                        # Doppelt zu einer Bodenaktion
                        if pitch == roll and roll == jump and jump == 0:
                            continue
                        # Handbremse für mögliche Wavedashes
                        handbrake = float(jump == 1 and (pitch != 0 or yaw != 0 or roll != 0))
                        actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])

    return np.array(actions, dtype=np.float32)


LOOKUP_TABLE = build_lookup_table()
NUM_ACTIONS = len(LOOKUP_TABLE)
