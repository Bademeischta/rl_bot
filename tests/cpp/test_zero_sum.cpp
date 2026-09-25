// Review-Befund R10: Das Experiment "team_spirit_01" misst in Wahrheit Zero-Sum im 1v1.
// Geprüft am echten Reward aus BuildLucyReward in einer echten Arena:
//  - team_spirit (tau) ist im 1v1 wirkungslos: jedes tau > 0 ergibt r_i - r_j
//  - mit goal/concede = 5 bleibt ein Tor nach dem Wrapper bei +-10 (wie baseline mit 10 ohne Wrapper)
//  - der Reward vor dem Wrapper (raw_step_reward) wird geloggt, weil die Summe nach dem Wrapper 0 ist
#include "test_util.h"

#include "env/cpp/Obs.h"
#include "env/cpp/Rewards.h"
#include "env/cpp/StateSetters.h"
#include "env/cpp/TimeoutCondition.h"
#include "train/cpp/Metrics.h"

#include <RLGymSim_CPP/Gym.h>
#include <RLGymSim_CPP/Math.h>
#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>

using namespace RLbot;

extern bool g_arenaReady;

namespace {

// Ball fliegt ins blaue Tor (Orange trifft), Autos weit weg und in fester Position.
class ShotOnBlueGoalSetter : public StateSetter {
public:
	virtual GameState ResetState(Arena* arena) {
		arena->ResetToRandomKickoff();
		BallState bs = {};
		bs.pos = Vec(0, -3500, 200);
		bs.vel = Vec(0, -4000, 0);
		arena->ball->SetState(bs);
		for (Car* car : arena->_cars) {
			CarState cs = {};
			cs.pos = Vec(car->team == Team::BLUE ? -2500.f : 2500.f, 3000, 17);
			cs.rotMat = RotMat::GetIdentity();
			car->SetState(cs);
		}
		return GameState(arena);
	}
};

RewardWeights OnlyGoals(float goal, float teamSpirit) {
	RewardWeights w = {};
	w.goal = goal; w.concede = goal;
	w.touchBallToGoalAccel = 0; w.offensivePotential = 0; w.distWeightedAlign = 0;
	w.velocityPlayerToBall = 0; w.saveBoost = 0; w.inAir = 0;
	w.teamSpirit = teamSpirit;
	return w;
}

// Spielt eine Episode bis zum Tor und liefert die Rewards je Team am Torschritt.
struct GoalRewards { float blue = 0, orange = 0; };
GoalRewards RewardsAtGoal(const RewardWeights& w) {
	auto* match = new Match(BuildLucyReward(w), { new GoalScoreCondition(), new TimeoutCondition(400) },
	                        new StackedPaddedOBS(3, 5, false), new DiscreteAction(), new ShotOnBlueGoalSetter(), 1, true);
	Gym gym(match, 8);
	gym.Reset();
	Gym::StepResult r = {};
	for (int i = 0; i < 400 && !r.done; i++)
		r = gym.Step(IList(2, 0));
	CHECK(r.done);
	CHECK(RLGSC::Math::IsBallScored(r.state.ball.pos));
	GoalRewards out;
	for (size_t p = 0; p < r.state.players.size(); p++)
		(r.state.players[p].team == Team::BLUE ? out.blue : out.orange) = r.reward[p];
	delete match;
	return out;
}

} // namespace

TEST(ZeroSum_1v1_Tor_bleibt_bei_10_wenn_goal_halbiert_ist) {
	if (!g_arenaReady) return;
	auto base = RewardsAtGoal(OnlyGoals(10.f, 0.f));    // baseline: 10, kein Wrapper
	CHECK_NEAR(base.orange, 10.0, 1e-5);
	CHECK_NEAR(base.blue, -10.0, 1e-5);
	auto zs = RewardsAtGoal(OnlyGoals(5.f, 0.1f));      // zero_sum.json: 5, Wrapper verdoppelt
	CHECK_NEAR(zs.orange, 10.0, 1e-5);
	CHECK_NEAR(zs.blue, -10.0, 1e-5);
	auto unhalved = RewardsAtGoal(OnlyGoals(10.f, 0.1f)); // ohne Halbierung wäre das Tor 20 wert
	CHECK_NEAR(unhalved.orange, 20.0, 1e-5);
}

TEST(ZeroSum_1v1_team_spirit_ist_wirkungslos) {
	if (!g_arenaReady) return;
	// Voller Lucy-Reward (dichtes Shaping), verschiedene tau: identische Rewards, Summe 0
	std::vector<std::vector<float>> perTau;
	for (float tau : { 0.1f, 0.5f, 1.0f }) {
		RewardWeights w = {};
		w.teamSpirit = tau;
		StateSetterWeights sw = {};
		sw.kickoff = 0; sw.random = 0; sw.aerial = 1;
		auto* match = new Match(BuildLucyReward(w), { new TimeoutCondition(1000) }, new StackedPaddedOBS(3, 5, false),
		                        new DiscreteAction(), new WeightedStateSetter(sw, 99), 1, true);
		Gym gym(match, 8);
		gym.Reset();
		std::vector<float> rewards;
		for (int i = 0; i < 30; i++) {
			auto r = gym.Step(IList{ i % 90, (i * 7) % 90 });
			CHECK_NEAR(r.reward[0] + r.reward[1], 0.0, 1e-5);
			rewards.push_back(r.reward[0]);
		}
		perTau.push_back(rewards);
		delete match;
	}
	for (size_t t = 1; t < perTau.size(); t++)
		for (size_t i = 0; i < perTau[0].size(); i++)
			CHECK_NEAR(perTau[t][i], perTau[0][i], 1e-5);
}

TEST(ZeroSum_roher_Reward_vor_dem_Wrapper_wird_geloggt) {
	if (!g_arenaReady) return;
	RewardWeights w = {};
	w.teamSpirit = 0.1f;
	StateSetterWeights sw = {};
	sw.kickoff = 0; sw.random = 0; sw.aerial = 1;
	auto* match = new Match(BuildLucyReward(w), { new TimeoutCondition(1000) }, new StackedPaddedOBS(3, 5, false),
	                        new DiscreteAction(), new WeightedStateSetter(sw, 7), 1, true);
	Gym gym(match, 8);
	gym.Reset();
	const RawRewardTap* tap = FindRawRewardTap(match->rewardFn);
	CHECK(tap != nullptr);

	RLGPC::Report metrics;
	double rawSum = 0, outSum = 0;
	for (int i = 0; i < 20; i++) {
		auto r = gym.Step(IList{ 3, 5 });
		CHECK_EQ(tap->lastRaw.size(), (size_t)2);
		// Ausgabe = r_i - r_j der rohen Rewards
		CHECK_NEAR(r.reward[0], tap->lastRaw[0] - tap->lastRaw[1], 1e-5);
		CHECK_NEAR(r.reward[1], tap->lastRaw[1] - tap->lastRaw[0], 1e-5);
		rawSum += tap->lastRaw[0] + tap->lastRaw[1];
		outSum += r.reward[0] + r.reward[1];
		AccumRawReward(match, metrics);
	}
	CHECK_NEAR(outSum, 0.0, 1e-4);            // das, was "Average Step Reward" sieht
	CHECK_GT(std::abs(rawSum), 1e-3);         // das dichte Shaping ist nicht 0
	RLGPC::Report agg;
	AggregateGameMetrics({ metrics }, agg);
	CHECK_NEAR(agg["raw_step_reward"], rawSum / 40.0, 1e-5);

	// Ohne Wrapper ist der Tap direkt der Reward, raw = Ausgabe
	RewardWeights plain = {};
	auto* fn = BuildLucyReward(plain);
	CHECK(FindRawRewardTap(fn) == fn);
	delete fn;
	delete match;
}
