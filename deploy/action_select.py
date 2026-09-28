"""Aktionsauswahl aus den Policy-Wahrscheinlichkeiten (Python-Gegenstück zu env/cpp/ActionSelect.h).

Die 90er-Aktionstabelle enthält viele Einträge, die in einer Lage dasselbe bewirken. Am Boden ohne
Sprung wirken nur Gas, Lenken, Boost und Handbremse: "Gas + Boost geradeaus" gibt es 9-mal,
"Gas ohne Boost geradeaus" einmal. argmax über Einträge wählt deshalb seltener Boost, als die
Policy will. argmax über Wirkungsklassen summiert erst die gleichwirkenden Einträge.

Wird von eval/kickoff_eval.py benutzt. Der RLBot-Bot (deploy/rlbot/bot.py) benutzt es NICHT;
eine Umstellung dort ist eine Deployment-Entscheidung.
"""
from __future__ import annotations

import numpy as np

from deploy.action_table import LOOKUP_TABLE

MODES = ("sample", "argmax", "argmax_group")


def effect_groups(table: np.ndarray, on_ground: bool) -> np.ndarray:
    """Klassennummer je Tabelleneintrag; gleiche Nummer = gleiche Wirkung in dieser Lage.

    Am Boden ohne Sprung: (Gas, Lenken, Boost, Handbremse). Am Boden mit Sprung: jeder Eintrag
    einzeln (nach dem Abheben wirken Nicken/Gieren/Rollen noch im selben Schritt). In der Luft:
    (Nicken, Gieren, Rollen, Sprung, Boost).
    """
    ids: dict[tuple, int] = {}
    out = np.empty(len(table), dtype=np.int64)
    for i, a in enumerate(table):
        jump = a[5] >= 0.5
        if on_ground and not jump:
            key = (0, a[0], a[1], a[6], a[7])
        elif on_ground:
            key = (1, i)
        else:
            key = (2, a[2], a[3], a[4], a[5], a[6])
        out[i] = ids.setdefault(tuple(float(x) for x in key), len(ids))
    return out


GROUND_GROUPS = effect_groups(LOOKUP_TABLE, True)
AIR_GROUPS = effect_groups(LOOKUP_TABLE, False)


def argmax_group(probs: np.ndarray, groups: np.ndarray) -> int:
    """Klasse mit der größten Summe, darin der wahrscheinlichste Eintrag."""
    sums = np.bincount(groups, weights=probs)
    best = int(np.argmax(sums))
    members = np.flatnonzero(groups == best)
    return int(members[np.argmax(probs[members])])


def select(probs: np.ndarray, mode: str, on_ground: bool, rng: np.random.Generator | None = None) -> int:
    if mode == "argmax":
        return int(np.argmax(probs))
    if mode == "argmax_group":
        return argmax_group(probs, GROUND_GROUPS if on_ground else AIR_GROUPS)
    if mode == "sample":
        rng = rng or np.random.default_rng()
        return int(rng.choice(len(probs), p=probs / probs.sum()))
    raise ValueError(f"Unbekannte Aktionsauswahl {mode!r} ({', '.join(MODES)})")
