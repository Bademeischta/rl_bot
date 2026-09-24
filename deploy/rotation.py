"""Euler-Winkel (pitch, yaw, roll) -> Richtungsvektoren, exakt wie in RocketSim.

RLBot liefert die Fahrzeugrotation als Euler-Winkel, das Obs-Layout braucht aber die
forward-/up-Vektoren. RocketSim rechnet über Bullets setEulerYPR(yaw, -pitch, -roll),
was intern setEulerZYX(-roll, -pitch, yaw) ist. Die RotMat-Zeilen (forward, right, up)
sind die SPALTEN der Bullet-Matrix (RocketSim ist spaltenweise, Bullet zeilenweise).

tests/test_rotation_parity.py prüft das gegen einen Dump aus dem C++-Code.
"""
from __future__ import annotations

import math

import numpy as np


def euler_to_rotmat(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Gibt (forward, right, up) als 3x3-Array zurück, je eine Zeile."""
    euler_x, euler_y, euler_z = -roll, -pitch, yaw

    ci, si = math.cos(euler_x), math.sin(euler_x)
    cj, sj = math.cos(euler_y), math.sin(euler_y)
    ch, sh = math.cos(euler_z), math.sin(euler_z)
    cc, cs, sc, ss = ci * ch, ci * sh, si * ch, si * sh

    # Bullet-Matrix, zeilenweise
    bullet = np.array([
        [cj * ch, sj * sc - cs, sj * cc + ss],
        [cj * sh, sj * ss + cc, sj * cs - sc],
        [-sj, cj * si, cj * ci],
    ])
    # RocketSim: rot[i][j] = bullet[j][i] -> forward/right/up sind die Spalten
    return bullet.T


def forward_up(pitch: float, yaw: float, roll: float) -> tuple[np.ndarray, np.ndarray]:
    mat = euler_to_rotmat(pitch, yaw, roll)
    return mat[0], mat[2]
