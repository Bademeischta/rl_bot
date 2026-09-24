import numpy as np
import pytest

from env.factory import build_default_env


@pytest.mark.parametrize("team_size", [1, 2, 3])
def test_env_runs_1000_steps(team_size):
    env = build_default_env(team_size=team_size, wrap_for_ppo=False)
    obs = env.reset()
    agents = list(obs.keys())
    assert len(agents) == 2 * team_size
    # zero_padding=3 -> Obs-Länge ist unabhängig von der Team-Größe
    obs_len = {len(o) for o in obs.values()}
    assert len(obs_len) == 1

    rng = np.random.default_rng(0)
    n_actions = env.action_spaces[agents[0]][1]
    for _ in range(1000):
        actions = {a: np.array([rng.integers(n_actions)]) for a in env.agents}
        obs, rew, term, trunc = env.step(actions)
        for o in obs.values():
            assert np.all(np.isfinite(o))
        if any(term.values()) or any(trunc.values()):
            obs = env.reset()
    env.close()


def test_obs_len_is_team_size_agnostic():
    lens = set()
    for ts in (1, 2, 3):
        env = build_default_env(team_size=ts, wrap_for_ppo=False)
        lens |= {len(o) for o in env.reset().values()}
        env.close()
    assert len(lens) == 1
