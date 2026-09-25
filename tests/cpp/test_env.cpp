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
