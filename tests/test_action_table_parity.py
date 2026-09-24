"""Die Aktionstabelle im Deployment muss exakt der aus dem Training entsprechen."""
from __future__ import annotations

import numpy as np

from deploy.action_table import LOOKUP_TABLE, NUM_ACTIONS
from tests.test_obs_parity import golden  # noqa: F401  (pytest-Fixture)


def test_action_count_is_90():
    assert NUM_ACTIONS == 90


def test_table_matches_cpp(golden):  # noqa: F811
    expected = np.array(golden["action_table"], dtype=np.float32)
    assert expected.shape == LOOKUP_TABLE.shape, (expected.shape, LOOKUP_TABLE.shape)
    mismatches = np.flatnonzero(np.abs(expected - LOOKUP_TABLE).max(axis=1) > 0)
    assert not len(mismatches), (
        f"Zeilen weichen ab: {mismatches[:5].tolist()}\n"
        f"C++:    {expected[mismatches[:3]]}\nPython: {LOOKUP_TABLE[mismatches[:3]]}"
    )


def test_no_duplicate_actions():
    unique = np.unique(LOOKUP_TABLE, axis=0)
    assert len(unique) == NUM_ACTIONS, "Tabelle enthält doppelte Aktionen"
