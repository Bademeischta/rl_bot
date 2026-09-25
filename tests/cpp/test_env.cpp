// Prüft ein echtes Trainings-Environment aus der EnvFactory (Audit N10, H6, K1).
// Braucht die Collision-Meshes; ohne sie werden diese Tests übersprungen.
#include "test_util.h"

#include "env/cpp/Obs.h"
#include "env/cpp/StateSetters.h"
#include "train/cpp/Config.h"
#include "train/cpp/EnvFactory.h"

#include <random>

using namespace RLGSC;
using namespace RLbot;

extern bool g_arenaReady;

static TrainConfig SmallConfig() {
	TrainConfig cfg = {};
	cfg.states = {};
	cfg.states.kickoff = 0; cfg.states.random = 0;
	cfg.states.aerial = 1; cfg.states.defense = 1;
	cfg.randomSeed = 321;
	return cfg;
}

TEST(EnvFactory_reicht_Seed_und_Shuffle_an_Env_weiter) {
	if (!g_arenaReady) return;
	TrainConfig cfg = SmallConfig();
	cfg.shuffleSlots = false;

	EnvFactory f1(cfg), f2(cfg);
	auto e1 = f1.Create();
	auto e2 = f2.Create();
	auto* s1 = dynamic_cast<WeightedStateSetter*>(e1.match->stateSetter);
	auto* s2 = dynamic_cast<WeightedStateSetter*>(e2.match->stateSetter);
	CHECK(s1 && s2);
	CHECK(s1->seed >= 0);
	CHECK_EQ(s1->seed, s2->seed);                       // gleicher Env-Index, gleicher Seed
	auto* o1 = dynamic_cast<StackedPaddedOBS*>(e1.match->obsBuilder);
	CHECK(o1);
	CHECK(!o1->shuffle);
	CHECK(o1->shuffleSeed >= 0);
	CHECK(o1->shuffleSeed != s1->seed);                 // eigener Strom für den Shuffle

	auto e1b = f1.Create();                             // zweites Env: anderer Seed
	auto* s1b = dynamic_cast<WeightedStateSetter*>(e1b.match->stateSetter);
	CHECK(s1b->seed != s1->seed);

	// Gleicher Seed -> gleicher Startzustand (nur eigene Szenen aktiv)
	auto obs1 = e1.gym->Reset();
	auto obs2 = e2.gym->Reset();
	CHECK_EQ(obs1.size(), obs2.size());
	for (size_t i = 0; i < obs1[0].size(); i++)
		CHECK_NEAR(obs1[0][i], obs2[0][i], 1e-6);

	cfg.seedEnvs = false;
	EnvFactory f3(cfg);
	auto e3 = f3.Create();
	CHECK_EQ(dynamic_cast<WeightedStateSetter*>(e3.match->stateSetter)->seed, (int64_t)-1);
	CHECK_EQ(dynamic_cast<StackedPaddedOBS*>(e3.match->obsBuilder)->shuffleSeed, (int64_t)-1);

	for (auto* e : { &e1, &e2, &e1b, &e3 }) { delete e->gym; delete e->match; }
}

TEST(EnvFactory_Env_laeuft_100_Schritte_und_Aktionsstack_wandert_um_eins) {
	if (!g_arenaReady) return;
	TrainConfig cfg = SmallConfig();
	cfg.shuffleSlots = false;
	cfg.gameTimeoutSecs = 1000;   // kein Timeout in 100 Schritten
	cfg.noTouchTimeoutSecs = 1000;
	EnvFactory factory(cfg);
	auto env = factory.Create();

	int stackStart = StackedPaddedOBS::BALL_FEATURES + CommonValues::BOOST_LOCATIONS_AMOUNT
		+ StackedPaddedOBS::PLAYER_FEATURES;
	int elems = (int)Action::ELEM_AMOUNT;
	int actionCount = env.match->actionParser->GetActionAmount();
	CHECK_EQ(actionCount, 90);

	auto obs = env.gym->Reset();
	CHECK_EQ((int)obs[0].size(), 257);
	std::mt19937 rng(5);
	for (int step = 0; step < 100; step++) {
		IList actions(env.match->playerAmount);
		for (auto& a : actions) a = (int)(rng() % actionCount);
		auto result = env.gym->Step(actions);
		if (result.done) {
			obs = env.gym->Reset();
			continue;
		}
		for (size_t p = 0; p < obs.size(); p++) {
			const FList& before = obs[p];
			const FList& after = result.obs[p];
			CHECK_EQ((int)after.size(), 257);
			// Audit K2/N10: Der Stack rückt pro Schritt um genau eine Aktion weiter:
			// die jüngsten 4 Aktionen von vorher sind die ältesten 4 von jetzt.
			for (int i = 0; i < 4 * elems; i++)
				CHECK_NEAR(after[stackStart + i], before[stackStart + elems + i], 1e-9);
			// und der jüngste Eintrag ist die gerade gespielte Aktion
			const Action& played = env.match->prevActions[p];
			for (int e = 0; e < elems; e++)
				CHECK_NEAR(after[stackStart + 4 * elems + e], played[e], 1e-6);
		}
		obs = result.obs;
	}
	delete env.gym;
	delete env.match;
}

// --- Truncation (Audit K1, Upstream-Patch) ------------------------------------

#include "env/cpp/TimeoutCondition.h"
#include "train/cpp/Metrics.h"
#include "RLGymPPO_CPP/Util/TorchFuncs.h"

#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>

TEST(K1_Upstream_Patch_ist_angewendet) {
	// Ohne den Patch wären Timeouts weiter Terminals; der Build muss ihn anwenden.
#ifdef RLGSC_HAS_TRUNCATION
	CHECK(true);
#else
	FAIL_AT("third_party/patches/rlgympppo_cpp_truncation.patch ist nicht angewendet");
#endif
}

#ifdef RLGSC_HAS_TRUNCATION

// Setzt den Ball so, dass er in Kürze ins blaue Tor fliegt; Autos weit weg.
class BallIntoBlueGoalSetter : public StateSetter {
public:
	virtual GameState ResetState(Arena* arena) {
		arena->ResetToRandomKickoff();
		BallState bs = {};
		bs.pos = Vec(0, -3500, 200);
		bs.vel = Vec(0, -4000, 0);
		arena->ball->SetState(bs);
		for (Car* car : arena->_cars) {
			CarState cs = {};
			cs.pos = Vec(car->team == Team::BLUE ? -3000.f : 3000.f, 4000, 17);
			cs.rotMat = RotMat::GetIdentity();
			car->SetState(cs);
		}
		return GameState(arena);
	}
};

static Match* MakeTruncationMatch(StateSetter* setter, int noTouchSteps, int timeoutSteps) {
	return new Match(new EventReward({}),
	                 { new NoTouchTruncation(noTouchSteps), new TimeoutCondition(timeoutSteps), new GoalScoreCondition() },
	                 new StackedPaddedOBS(3, 5, false), new DiscreteAction(), setter, 1, true);
}

// Schrittet mit Nullaktionen bis done; liefert das letzte StepResult.
static Gym::StepResult StepUntilDone(Gym& gym, int maxSteps) {
	gym.Reset();
	Gym::StepResult result = {};
	for (int i = 0; i < maxSteps; i++) {
		result = gym.Step(IList(gym.match->playerAmount, 0));
		if (result.done)
			return result;
	}
	FAIL_AT("Episode endete nicht innerhalb von " << maxSteps << " Schritten");
	return result;
}

TEST(K1_Spielzeit_Timeout_ist_done_und_truncated) {
	if (!g_arenaReady) return;
	Match* match = MakeTruncationMatch(new KickoffSetter(), 100000, 12);
	Gym gym(match, 8);
	auto result = StepUntilDone(gym, 50);
	CHECK(result.done);
	CHECK(result.truncated);
	auto end = ClassifyEpisodeEnd(result.state, match, result.truncated);
	CHECK(end.timeLimit);
	CHECK(!end.goal);
	CHECK(end.truncated);
	delete match;
}

TEST(K1_NoTouch_Timeout_ist_done_und_truncated) {
	if (!g_arenaReady) return;
	Match* match = MakeTruncationMatch(new KickoffSetter(), 9, 100000);
	Gym gym(match, 8);
	auto result = StepUntilDone(gym, 50);
	CHECK(result.done);
	CHECK(result.truncated);
	auto end = ClassifyEpisodeEnd(result.state, match, result.truncated);
	CHECK(end.noTouch);
	CHECK(!end.goal);
	delete match;
}

TEST(K1_Tor_ist_done_aber_nicht_truncated) {
	if (!g_arenaReady) return;
	Match* match = MakeTruncationMatch(new BallIntoBlueGoalSetter(), 100000, 100000);
	Gym gym(match, 8);
	auto result = StepUntilDone(gym, 200);
	CHECK(result.done);
	CHECK(!result.truncated);
	auto end = ClassifyEpisodeEnd(result.state, match, result.truncated);
	CHECK(end.goal);
	CHECK(!end.truncated);
	delete match;
}

TEST(K1_Tor_und_Timeout_im_selben_Schritt_zaehlt_als_Tor) {
	// Timeout genau in dem Schritt, in dem der Ball die Torlinie überquert: nicht truncated.
	if (!g_arenaReady) return;
	int goalStep = -1;
	{
		Match* probe = MakeTruncationMatch(new BallIntoBlueGoalSetter(), 100000, 100000);
		Gym gym(probe, 8);
		gym.Reset();
		for (int i = 1; i <= 200 && goalStep < 0; i++)
			if (gym.Step(IList(2, 0)).done) goalStep = i;
		delete probe;
	}
	CHECK_GT(goalStep, 0);
	Match* match = MakeTruncationMatch(new BallIntoBlueGoalSetter(), 100000, goalStep);
	Gym gym(match, 8);
	auto result = StepUntilDone(gym, 200);
	CHECK(result.done);
	CHECK(!result.truncated);
	delete match;
}

TEST(K1_GAE_bootstrappt_Truncation_und_nicht_Terminal) {
	using RLGPC::TorchFuncs::ComputeGAE;
	const float gamma = 0.9f, lambda = 0.95f;
	// 4 Schritte, Reward 0, alle Wertschätzungen 1 (values hat einen Eintrag mehr: Endzustand)
	FList rews = { 0, 0, 0, 0 };
	FList values = { 1, 1, 1, 1, 1 };
	torch::Tensor adv, valTargets;
	FList returns;

	{ // Terminal (Tor) in Schritt 2: kein Bootstrap, Ziel = 0
		FList dones = { 0, 0, 1, 0 }, trunc = { 0, 0, 0, 0 };
		ComputeGAE(rews, dones, trunc, values, adv, valTargets, returns, gamma, lambda, 0, 0);
		CHECK_NEAR(adv[2].item<float>(), 0.0 - 1.0, 1e-6);
		// Schritt 1 sieht Schritt 2 (kein Abbruch der Kette vor einem Terminal)
		CHECK_NEAR(adv[1].item<float>(), (gamma * 1 - 1) + gamma * lambda * (-1.0), 1e-6);
	}
	{ // Truncation (Timeout) in Schritt 2 mit bekanntem Folgewert 5: Bootstrap mit gamma*5
		FList dones = { 0, 0, 0, 0 }, trunc = { 0, 0, 1, 0 };
		FList truncNext = { 0, 0, 5, 0 };
		ComputeGAE(rews, dones, trunc, values, adv, valTargets, returns, gamma, lambda, 0, 0, truncNext);
		// adv[2] enthält nichts von Schritt 3 (neue Episode): die Kette bricht HINTER der Truncation
		CHECK_NEAR(adv[2].item<float>(), gamma * 5 - 1, 1e-6);
		// Schritt 1 -> 2 ist ein echter Übergang derselben Episode: adv[1] sieht adv[2]
		CHECK_NEAR(adv[1].item<float>(), (gamma * 1 - 1) + gamma * lambda * (gamma * 5 - 1), 1e-6);
		CHECK_NEAR(returns[2], 0.0, 1e-6);
		// Schritt 3 (neue Episode) bleibt unberührt
		CHECK_NEAR(adv[3].item<float>(), gamma * 1 - 1, 1e-6);
	}
	{ // Ohne Folgewerte (altes Verhalten): Bootstrap vom nächsten Listeneintrag (Wert 1)
		FList dones = { 0, 0, 0, 0 }, trunc = { 0, 0, 1, 0 };
		ComputeGAE(rews, dones, trunc, values, adv, valTargets, returns, gamma, lambda, 0, 0);
		CHECK_NEAR(adv[2].item<float>(), gamma * 1 - 1, 1e-6);
	}
}

#endif // RLGSC_HAS_TRUNCATION
