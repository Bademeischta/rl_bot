"""Die Euler->Vektor-Umrechnung im Deployment muss exakt der aus RocketSim entsprechen."""
from __future__ import annotations

import numpy as np

from deploy.rotation import euler_to_rotmat, forward_up
from tests.test_obs_parity import golden  # noqa: F401  (pytest-Fixture)


def test_matches_cpp(golden):  # noqa: F811
    worst = 0.0
    for case in golden["rotation_cases"]:
        mat = euler_to_rotmat(case["pitch"], case["yaw"], case["roll"])
        for row, key in enumerate(("forward", "right", "up")):
            expected = np.array(case[key])
            diff = float(np.abs(mat[row] - expected).max())
            worst = max(worst, diff)
            assert diff < 1e-5, (
                f"{key} weicht ab bei pitch={case['pitch']:.3f} yaw={case['yaw']:.3f} "
                f"roll={case['roll']:.3f}: C++ {expected} vs Python {mat[row]}"
            )
    print(f"\n{len(golden['rotation_cases'])} Rotationen geprüft, größte Abweichung: {worst:.3e}")


def test_identity_rotation():
    fwd, up = forward_up(0.0, 0.0, 0.0)
    assert np.allclose(fwd, [1, 0, 0], atol=1e-9)
    assert np.allclose(up, [0, 0, 1], atol=1e-9)


def test_rotmat_is_orthonormal(golden):  # noqa: F811
    for case in golden["rotation_cases"]:
        mat = euler_to_rotmat(case["pitch"], case["yaw"], case["roll"])
        assert np.allclose(mat @ mat.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(mat), 1.0, atol=1e-9)
