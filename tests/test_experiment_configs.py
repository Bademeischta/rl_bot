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


def test_main_run_config_is_main_config_plus_confirmed_winners_only():
    """Hauptlauf-Config seit Stufe 3 (AUDIT.md §7.9): lucy_1v1.json plus die bestätigten Gewinner (nur
    zero_sum), gleicher Checkpoint-Ordner, Obs/Aktionen unverändert. Dazu checkpoints_to_keep 50
    (Nutzer, 28.09.2026), keine Verhaltensänderung: Ältere Checkpoints bleiben als Duell-Gegner."""
    main = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1.json").read_text(encoding="utf-8")))
    config = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1_zero_sum.json").read_text(encoding="utf-8")))
    assert diff(main, config) == {**EXPECTED["zero_sum"], "learner.checkpoints_to_keep": (10, 50)}
    assert config["learner.checkpoint_folder"] == "runs/lucy_1v1/checkpoints"
    assert config["env.max_players"] == 3 and config["env.action_stack_size"] == 5
    assert "checkpoints_to_keep" in config["_comment"]


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


# --- Spieltest, Phase C (AUDIT.md §8): je genau eine Änderung gegenüber zero_sum.json -------------

SPIELTEST_EXPECTED = {
    "sp_kickoff_drill": {"state_setters.kickoff_drill": (None, 4.0)},
    "sp_kickoff_first_touch": {"rewards.kickoff_first_touch": (None, 2.0)},
    "sp_potential_shaping": {"rewards.potential_shaping_scale": (None, 8.0)},
    # symmetrischer Torwert: goal und concede gemeinsam (wie in zero_sum.json)
    "sp_goal_x3": {"rewards.goal": (5.0, 15.0), "rewards.concede": (5.0, 15.0)},
    "sp_air_touch": {"rewards.air_touch": (None, 3.0)},
    "sp_aerial_share": {"state_setters.aerial": (0.5, 2.0)},
    "sp_no_in_air": {"rewards.in_air": (0.02, 0.0)},
    "sp_mode_2v2": {"env.mode_mix": ([1.0, 0.0, 0.0], [3.0, 1.0, 0.0])},
}


@pytest.mark.parametrize("name,expected", list(SPIELTEST_EXPECTED.items()))
def test_spieltest_experiment_changes_exactly_one_thing_against_zero_sum(name, expected):
    assert diff(load("zero_sum"), load(name)) == expected
    assert load(name)["metrics.run"] == name
    assert "NICHT gestartet" in load(name)["_comment"]


def test_team_spirit_experiment_changes_only_tau_against_the_2v2_run():
    assert diff(load("sp_mode_2v2"), load("sp_mode_2v2_tau05")) == {"rewards.team_spirit": (0.1, 0.5)}


# --- Spieltest: Hauptlauf-Vorschläge (AUDIT.md §8.10), nicht gestartet ------------------------

def test_main_run_proposals_add_only_the_kept_changes():
    main = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1_zero_sum.json").read_text(encoding="utf-8")))
    one = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1_zero_sum_drill.json").read_text(encoding="utf-8")))
    team = flatten(json.loads((ROOT / "train" / "configs" / "lucy_team_zero_sum.json").read_text(encoding="utf-8")))
    assert diff(main, one) == {"state_setters.kickoff_drill": (None, 4.0)}
    assert diff(one, team) == {"env.mode_mix": ([1.0, 0.0, 0.0], [3.0, 1.0, 0.0]), "rewards.team_spirit": (0.1, 0.5),
                               "learner.checkpoint_folder": ("runs/lucy_1v1/checkpoints", "runs/lucy_team/checkpoints")}
    for cfg in (one, team):
        assert cfg["env.max_players"] == 3 and cfg["env.action_stack_size"] == 5
        assert cfg["learner.policy_layer_sizes"] == [512, 512, 512]
        assert "NICHT gestartet" in cfg["_comment"]


# --- Geschwindigkeit (AUDIT.md §9): Lernvergleich, nur Learner-Schalter gegenüber sp_kickoff_drill ---

SPEED_OVERLAP = {"learner.collection_during_learn": (False, True), "learner.exp_buffer_on_device": (None, True),
                 "learner.infer_during_learn": (None, True), "learner.collect_limit_factor": (None, 1.0),
                 "learner.learner_high_priority_stream": (None, True)}
SPEED_EXPECTED = {
    "speed_overlap": SPEED_OVERLAP,
    "speed_overlap_amp": {**SPEED_OVERLAP, "learner.autocast_learn": (None, True)},
}


@pytest.mark.parametrize("name,expected", list(SPEED_EXPECTED.items()))
def test_speed_experiments_change_only_speed_switches_against_the_drill_run(name, expected):
    """Referenz ist sp_kickoff_drill (= Hauptlauf-Config lucy_1v1_zero_sum_drill als Experiment, Phase C);
    Rewards, Szenen, Netz und PPO-Hyperparameter bleiben gleich."""
    assert diff(load("sp_kickoff_drill"), load(name)) == expected
    assert load(name)["metrics.run"] == name
    assert "NICHT gestartet" in load(name)["_comment"]


def test_fast_main_run_proposal_adds_only_speed_switches():
    """Hauptlauf-Config Geschwindigkeit (AUDIT.md §9.6): lucy_1v1_zero_sum_drill.json plus nur die
    Learner-Schalter aus speed_overlap_amp und die Checkpoint-/Skill-Tracker-Abstände (B1, AUDIT.md §10);
    gleicher Checkpoint-Ordner, Obs/Aktionen/Netze/Rewards/PPO unverändert."""
    drill = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1_zero_sum_drill.json").read_text(encoding="utf-8")))
    fast = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1_zero_sum_drill_fast.json").read_text(encoding="utf-8")))
    assert diff(drill, fast) == {**SPEED_EXPECTED["speed_overlap_amp"], **HISTORY_EXPECTED}
    assert fast["learner.checkpoint_folder"] == "runs/lucy_1v1/checkpoints"
    assert fast["env.max_players"] == 3 and fast["env.action_stack_size"] == 5
    assert fast["learner.policy_layer_sizes"] == [512, 512, 512]
    assert "Hauptlauf-Config" in fast["_comment"]


# B1 (Hauptlauf-Betrieb, 30.09.2026): Bei ~180.000 SPS deckten 50 Checkpoints à 25 Mio. Steps nur ~2 h ab,
# und der Skill-Tracker (500 Mio. Abstand) fand beim Neustart 3 von 20 alten Versionen.
HISTORY_EXPECTED = {
    "learner.timesteps_per_save": (25_000_000, 50_000_000),
    "learner.checkpoints_to_keep": (50, 200),
    "metrics.skill_timesteps_per_version": (500_000_000, 250_000_000),
}
CHECKPOINT_MB = 15.63           # gemessen: runs/lucy_1v1/checkpoints/6037692544 (Policy, Critic, 2 Optimizer, Stats)
SPS = 180_000


def test_main_run_keeps_billions_of_steps_history_within_the_disk_budget():
    fast = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1_zero_sum_drill_fast.json").read_text(encoding="utf-8")))
    save, keep = fast["learner.timesteps_per_save"], fast["learner.checkpoints_to_keep"]
    history = save * keep
    assert history >= 5_000_000_000                                # mehrere Milliarden Steps
    assert history / SPS / 3600 >= 8                               # mindestens eine Nacht bei ~180.000 SPS
    assert keep * CHECKPOINT_MB / 1024 <= 10                       # Platte: ~35 GB frei, 10 GB Reserve, Rest Luft
    assert save / SPS / 60 <= 5                                    # höchstens ~5 min Fortschritt bei einem Absturz
    # Skill-Tracker: alle Versionen liegen in der Historie (Learner::Load sucht sie dort) ...
    per_version, versions = fast["metrics.skill_timesteps_per_version"], fast["metrics.skill_max_versions"]
    assert per_version * versions <= history
    assert per_version % save == 0                                 # ... und fallen auf gespeicherte Checkpoints
    # Regressions-Check (tools/regression_check.py): Gegner ~1 Mrd. Steps älter muss vorhanden sein
    assert history >= 2 * 1_000_000_000


# --- Spieltest 2 (01.10.2026, AUDIT.md §11): je genau eine Änderung gegenüber der aktuellen Hauptlauf-Config ---

SPIELTEST2_EXPECTED = {
    "sp2_kickoff_first_touch": {"rewards.kickoff_first_touch": (None, 2.0)},
    "sp2_aerial_share": {"state_setters.aerial": (0.5, 2.0)},
    # zweite Runde (§11.5): save_boost als vermutete Ursache des Anstoßes ohne Boost
    "sp2_save_boost_01": {"rewards.save_boost": (0.3, 0.1)},
    # Bündel: beide Anstoß-Hebel zusammen
    "sp2_save_boost_01_kickoff": {"rewards.save_boost": (0.3, 0.1), "rewards.kickoff_first_touch": (None, 2.0)},
}


def test_spieltest2_reference_is_the_running_main_config_as_an_experiment():
    """Nur Experiment-Buchhaltung weicht ab: Seeds, Ordner, Checkpoint-Abstand, Skill-Tracker-Abstand."""
    fast = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1_zero_sum_drill_fast.json").read_text(encoding="utf-8")))
    assert diff(fast, load("sp2_reference")) == {
        "env.seed_envs": (None, True),
        "learner.checkpoint_folder": ("runs/lucy_1v1/checkpoints", "runs/EXPERIMENT/checkpoints"),
        "learner.checkpoints_to_keep": (200, 10), "learner.timesteps_per_save": (50_000_000, 25_000_000),
        "metrics.group": ("phase3", "experiments"), "metrics.skill_timesteps_per_version": (250_000_000, 500_000_000),
    }
    assert load("sp2_reference")["metrics.run"] == "sp2_reference"


@pytest.mark.parametrize("name,expected", list(SPIELTEST2_EXPECTED.items()))
def test_spieltest2_experiment_changes_only_the_named_values_against_the_reference(name, expected):
    assert diff(load("sp2_reference"), load(name)) == expected
    assert load(name)["metrics.run"] == name
    assert load(name)["env.max_players"] == 3 and load(name)["env.action_stack_size"] == 5
    assert load(name)["learner.policy_layer_sizes"] == [512, 512, 512]


def test_kickoff_main_run_proposal_adds_only_the_two_kickoff_values():
    """Vorschlag nach Spieltest 2 (AUDIT.md §11.5): laufende Hauptlauf-Config plus genau das Bündel aus
    sp2_save_boost_01_kickoff; gleicher Checkpoint-Ordner, Obs/Aktionen/Netze/PPO unverändert."""
    fast = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1_zero_sum_drill_fast.json").read_text(encoding="utf-8")))
    kick = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1_zero_sum_drill_fast_kickoff.json").read_text(encoding="utf-8")))
    assert diff(fast, kick) == SPIELTEST2_EXPECTED["sp2_save_boost_01_kickoff"]
    assert kick["learner.checkpoint_folder"] == "runs/lucy_1v1/checkpoints"
    assert kick["env.max_players"] == 3 and kick["env.action_stack_size"] == 5
    assert kick["learner.policy_layer_sizes"] == [512, 512, 512]
    assert "NICHT gestartet ohne OK" in kick["_comment"]


def test_lr1e4_main_run_halves_the_learning_rates_in_its_own_folder():
    """Hauptlauf ab 03.10.2026 (AUDIT.md §11.7): Anstoß-Config mit halbierter Lernrate, eigener Ordner, damit
    runs/lucy_1v1 unverändert bleibt; Obs/Aktionen/Netze/Rewards/übrige PPO-Werte gleich."""
    kick = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1_zero_sum_drill_fast_kickoff.json").read_text(encoding="utf-8")))
    lr = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1_kickoff_lr1e4.json").read_text(encoding="utf-8")))
    assert diff(kick, lr) == {
        "learner.policy_lr": (0.0002, 0.0001),
        "learner.critic_lr": (0.0002, 0.0001),
        "learner.checkpoint_folder": ("runs/lucy_1v1/checkpoints", "runs/lucy_1v1_lr1e4/checkpoints"),
    }
    assert lr["env.max_players"] == 3 and lr["env.action_stack_size"] == 5
    assert lr["learner.policy_layer_sizes"] == [512, 512, 512]


# --- Luftspiel (04.10.2026, AUDIT.md §11.8): gegenüber der laufenden Hauptlauf-Config lucy_1v1_kickoff_lr1e4 ---

SP3_EXPECTED = {
    "sp3_air_touch": {"rewards.air_touch": (None, 3.0)},
    # Bündel: Reward plus mehr Gelegenheiten
    "sp3_air_touch_aerial": {"rewards.air_touch": (None, 3.0), "state_setters.aerial": (0.5, 2.0)},
}


def test_sp3_reference_is_the_running_main_config_as_a_1g_experiment():
    main = flatten(json.loads((ROOT / "train" / "configs" / "lucy_1v1_kickoff_lr1e4.json").read_text(encoding="utf-8")))
    assert diff(main, load("sp3_reference")) == {
        "env.seed_envs": (None, True),
        "learner.checkpoint_folder": ("runs/lucy_1v1_lr1e4/checkpoints", "runs/EXPERIMENT/checkpoints"),
        "learner.timesteps_per_save": (50_000_000, 100_000_000),
        "learner.checkpoints_to_keep": (200, 12),
        "metrics.group": ("phase3", "experiments"),
        "metrics.skill_timesteps_per_version": (250_000_000, 500_000_000),
    }
    ref = load("sp3_reference")
    # 1 Mrd. Steps: alle Zwischenstände im Abstand von 100 Mio. bleiben erhalten
    assert ref["learner.timesteps_per_save"] * (ref["learner.checkpoints_to_keep"] - 2) >= 1_000_000_000
    assert ref["learner.policy_lr"] == ref["learner.critic_lr"] == 0.0001


@pytest.mark.parametrize("name,expected", list(SP3_EXPECTED.items()))
def test_sp3_experiment_changes_only_the_named_values_against_the_reference(name, expected):
    assert diff(load("sp3_reference"), load(name)) == expected
    assert load(name)["metrics.run"] == name
    assert load(name)["env.max_players"] == 3 and load(name)["env.action_stack_size"] == 5
    assert load(name)["learner.policy_layer_sizes"] == [512, 512, 512]
