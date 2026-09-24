"""RLGym-v2-Environments. Phase 0: Default-Setup analog rlgym-ppo/rlgym_v2_example.py (nur für SPS-Messung)."""
import numpy as np

TICK_SKIP = 8
GAME_TICK_RATE = 120


def build_default_env(team_size: int = 1, wrap_for_ppo: bool = True):
    from rlgym.api import RLGym
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (AnyCondition, GoalCondition,
                                                      NoTouchTimeoutCondition, TimeoutCondition)
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (FixedTeamSizeMutator, KickoffMutator,
                                                     MutatorSequence)

    env = RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=team_size, orange_size=team_size),
            KickoffMutator(),
        ),
        obs_builder=DefaultObs(
            zero_padding=3,
            pos_coef=np.asarray([1 / common_values.SIDE_WALL_X,
                                 1 / common_values.BACK_NET_Y,
                                 1 / common_values.CEILING_Z]),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
            boost_coef=1 / 100.0,
        ),
        action_parser=RepeatAction(LookupTableAction(), repeats=TICK_SKIP),
        reward_fn=CombinedReward((GoalReward(), 10.0), (TouchReward(), 0.1)),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(timeout_seconds=30.0),
            TimeoutCondition(timeout_seconds=300.0),
        ),
        transition_engine=RocketSimEngine(),
    )
    if not wrap_for_ppo:
        return env
    from rlgym_ppo.util import RLGymV2GymWrapper
    return RLGymV2GymWrapper(env)
