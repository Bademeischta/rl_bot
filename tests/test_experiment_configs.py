"""Die Stufe-3-Experiment-Configs ändern gegenüber der Baseline genau EINE Sache."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "train" / "configs" / "experiments"
IGNORE = {"_comment", "metrics.run"}


def flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def load(name: str) -> dict:
    return flatten(json.loads((EXP / f"{name}.json").read_text(encoding="utf-8")))


def diff(a: dict, b: dict) -> dict[str, tuple]:
    keys = (set(a) | set(b)) - IGNORE
    return {k: (a.get(k), b.get(k)) for k in sorted(keys) if a.get(k) != b.get(k)}


EXPECTED = {
    "h2_ent_coef_0004": {"learner.ent_coef": (0.01, 0.004)},
    "h3_no_shuffle": {"env.shuffle_slots": (None, False)},
    # Review R10: Zero-Sum im 1v1 (tau wirkungslos), Tor/Gegentor halbiert -> nach dem Wrapper +-10
    "zero_sum": {"rewards.team_spirit": (0.0, 0.1), "rewards.goal": (10.0, 5.0), "rewards.concede": (10.0, 5.0)},
}


def test_zero_sum_replaces_team_spirit_01_and_says_what_it_measures():
    assert not (EXP / "team_spirit_01.json").exists()
    raw = json.loads((EXP / "zero_sum.json").read_text(encoding="utf-8"))
    assert raw["metrics"]["run"] == "zero_sum"
    comment = raw["_comment"]
    assert "WIRKUNGSLOS" in comment and "r_i - r_j" in comment and "raw_step_reward" in comment


def test_baseline_equals_main_config_except_run_and_folder():
    main = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1.json").read_text(encoding="utf-8")))
    d = diff(main, load("baseline"))
    # seed_envs: Default false (altes Verhalten, R6), Experimente setzen es ausdrücklich
    assert set(d) == {"learner.checkpoint_folder", "metrics.group", "env.seed_envs"}, d
    assert d["env.seed_envs"] == (None, True)
    assert load("baseline")["env.game_timeout_secs"] == 900.0          # K1a ist drin


def test_every_experiment_config_seeds_its_envs_explicitly():
    """Review-Befund R6: Default ist false (Hauptlauf unverändert), Experimente brauchen true."""
    configs = sorted(EXP.glob("*.json"))
    assert len(configs) >= 8
    for path in configs:
        assert flatten(json.loads(path.read_text(encoding="utf-8"))).get("env.seed_envs") is True, path.name


@pytest.mark.parametrize("name,expected", list(EXPECTED.items()))
def test_each_experiment_changes_exactly_one_thing(name, expected):
    d = diff(load("baseline"), load(name))
    assert d == expected, d


def test_k3_changes_only_the_rewards_block():
    d = diff(load("baseline"), load("k3_rewards"))
    assert d and all(k.startswith("rewards.") for k in d), d
    k3 = load("k3_rewards")
    assert k3["rewards.goal"] == 50.0 and k3["rewards.concede"] == 50.0
    assert k3["rewards.in_air"] == 0.0 and k3["rewards.save_boost"] == 0.05
    assert k3["rewards.offensive_potential_krc"] == 0.3 and k3["rewards.dist_weighted_align_krc"] == 0.1
    assert k3["rewards.velocity_player_to_ball"] == 0.03 and k3["rewards.touch_ball_to_goal_accel"] == 1.0
    assert k3["rewards.team_spirit"] == 0.0


def test_all_experiments_keep_obs_layout_and_actions():
    for path in EXP.glob("*.json"):
        cfg = flatten(json.loads(path.read_text(encoding="utf-8")))
        assert cfg["env.max_players"] == 3 and cfg["env.action_stack_size"] == 5, path.name
        assert cfg["learner.checkpoint_folder"] == "runs/EXPERIMENT/checkpoints", path.name


def test_main_run_proposal_is_main_config_plus_confirmed_winners_only():
    """Stufe 3, Schritt 3: Vorschlag für den fortgesetzten Hauptlauf = lucy_1v1.json plus die
    bestätigten Gewinner (nur zero_sum, AUDIT.md §7.9), gleicher Checkpoint-Ordner, Obs/Aktionen
    unverändert. Nicht gestartet."""
    main = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1.json").read_text(encoding="utf-8")))
    proposal = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1_zero_sum.json").read_text(encoding="utf-8")))
    assert diff(main, proposal) == EXPECTED["zero_sum"]
    assert proposal["learner.checkpoint_folder"] == "runs/lucy_1v1/checkpoints"
    assert proposal["env.max_players"] == 3 and proposal["env.action_stack_size"] == 5
    assert "NICHT gestartet" in proposal["_comment"]


def test_proposed_extension_tests_h3_on_top_of_zero_sum_with_one_change():
    """Vorgeschlagene Verlängerung (AUDIT.md §7.9): H3 auf Zero-Sum-Basis, genau eine Änderung."""
    assert diff(load("zero_sum"), load("zero_sum_h3")) == {"env.shuffle_slots": (None, False)}
    assert load("zero_sum_h3")["metrics.run"] == "zero_sum_h3"


# --- Stufe 4, H5: Gradientenschritte pro Iteration ------------------------------

H5_EXPECTED = {
    "h5_updates6_epochs2_buf3": (2, 3),
    "h5_updates3_epochs1_buf3": (1, 3),
    "h5_updates2_epochs2_buf1": (2, 1),
}


@pytest.mark.parametrize("name,expected", list(H5_EXPECTED.items()))
def test_h5_configs_change_only_epochs_and_buffer(name, expected):
    cfg = load(name)
    d = diff(load("baseline"), cfg)
    assert set(d) <= {"learner.ppo_epochs", "learner.exp_buffer_iterations"}, d
    assert (cfg["learner.ppo_epochs"], cfg["learner.exp_buffer_iterations"]) == expected
    # Batch = Iteration: Gradientenschritte je Iteration = epochs * buffer
    assert cfg["learner.ppo_batch_size"] == cfg["learner.timesteps_per_iteration"]
    updates = expected[0] * expected[1]
    assert str(updates) in name
