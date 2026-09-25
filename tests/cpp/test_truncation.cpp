// Review-Befund R4 (K1b): Timeouts müssen vom Wert der LETZTEN Beobachtung der Episode
// bootstrappen, nicht von der Reset-Beobachtung der nächsten Episode.
//
// Die erste Fassung des Truncation-Patches las in ThreadAgent stepResult.obs, das GameInst::Step
// zu diesem Zeitpunkt aber schon mit gym->Reset() überschrieben hatte. Die Tests hier gehen
// deshalb über den echten Pfad Gym -> GameInst -> ThreadAgent -> Learner/GAE und prüfen nicht nur
// die GAE-Formel mit handgebauten Eingaben (das tat K1_GAE_bootstrappt_Truncation_und_nicht_Terminal
// in test_env.cpp, der den Fehler deshalb nicht bemerkt hat).
#include "test_util.h"

#include "env/cpp/Obs.h"
#include "env/cpp/StateSetters.h"
#include "env/cpp/TimeoutCondition.h"
#include "train/cpp/Config.h"
#include "train/cpp/EnvFactory.h"

#include <RLGymPPO_CPP/Learner.h>
#include <RLGymPPO_CPP/Threading/GameInst.h>
#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>

#include <algorithm>
#include <fstream>
#include <mutex>

using namespace RLbot;

extern bool g_arenaReady;

#ifdef RLGSC_HAS_FINAL_OBS

// Kickoff-Start, NoTouch-Timeout nach wenigen Schritten: Vom Anstoßpunkt aus erreicht in
// noTouchSteps * 8 Ticks kein Auto den Ball, also endet jede Episode per Truncation.
static RLGPC::EnvCreateResult MakeNoTouchEnv(int noTouchSteps, bool asTruncation = true) {
	auto* match = new Match(new EventReward({}),
	                        { new NoTouchTruncation(noTouchSteps, asTruncation), new TimeoutCondition(100000, asTruncation),
	                          new GoalScoreCondition() },
	                        new StackedPaddedOBS(3, 5, false), new DiscreteAction(), new KickoffSetter(), 1, true);
	return { match, new Gym(match, 8) };
}

TEST(K1b_GameInst_legt_letzte_Obs_vor_dem_Reset_ab) {
	if (!g_arenaReady) return;
	auto env = MakeNoTouchEnv(4);
	RLGPC::GameInst game(env.gym, env.match);

	FList2 obsAtDone;   // was der Step-Callback (läuft vor dem Reset) als Beobachtung sieht
	game.stepCallback = [&](RLGPC::GameInst*, const Gym::StepResult& r, RLGPC::Report&) {
		if (r.done) obsAtDone = r.obs;
	};
	game.Start();
	Gym::StepResult result = {};
	int steps = 0;
	do {
		result = game.Step(IList(2, 0));
		steps++;
	} while (!result.done && steps < 20);

	CHECK(result.done);
	CHECK(result.truncated);
	CHECK_EQ(steps, 4);
	// finalObs = letzte Beobachtung der Episode, obs = Reset-Beobachtung der nächsten
	CHECK_EQ(result.finalObs.size(), (size_t)2);
	CHECK(result.finalObs == obsAtDone);
	CHECK(result.obs == game.curObs);
	CHECK(result.obs != result.finalObs);
	// Ohne Episodenende bleibt finalObs leer (keine Kopie im Normalfall)
	auto next = game.Step(IList(2, 0));
	CHECK(!next.done);
	CHECK(next.finalObs.empty());
}

TEST(K1b_Echter_Pfad_Timeout_bootstrappt_von_letzter_Obs_nicht_von_Reset_Obs) {
	if (!g_arenaReady) return;

	RLGPC::LearnerConfig lc = {};
	lc.numThreads = 1;
	lc.numGamesPerThread = 2;
	lc.timestepsPerIteration = 256;
	lc.timestepLimit = 256;              // genau eine Iteration
	lc.expBufferSize = 512;
	lc.ppo.batchSize = 256;
	lc.ppo.miniBatchSize = 128;
	lc.ppo.epochs = 1;
	lc.ppo.policyLayerSizes = { 16 };
	lc.ppo.criticLayerSizes = { 16 };
	lc.gaeGamma = 0.9f;
	lc.deviceType = RLGPC::LearnerDeviceType::CPU;
	lc.checkpointLoadFolder = "";
	lc.checkpointSaveFolder = "";
	lc.sendMetrics = false;
	lc.randomSeed = 7;

	std::mutex mutex;
	std::vector<FList> finalObsSeen;     // aus dem Step-Callback, also VOR dem Reset
	RLGPC::TruncationDiagnostics diag = {};
	bool gotDiag = false;
	RLGPC::Report report = {};

	{
		RLGPC::Learner learner([]() { return MakeNoTouchEnv(4); }, lc);
		learner.stepCallback = [&](RLGPC::GameInst*, const Gym::StepResult& r, RLGPC::Report&) {
			if (!r.done || !r.truncated) return;
			std::lock_guard<std::mutex> lock(mutex);
			for (auto& o : r.obs) finalObsSeen.push_back(o);
		};
		learner.truncationDiagnosticsCallback = [&](const RLGPC::TruncationDiagnostics& d) {
			diag = d;
			gotDiag = true;
		};
		learner.iterationCallback = [&](RLGPC::Learner*, RLGPC::Report& r) { report = r; };
		learner.Learn();
	}

	CHECK(gotDiag);
	size_t n = diag.envTruncIdx.size();
	CHECK_GT(n, 8);
	CHECK_EQ((double)report["Timeout Truncations"], (double)n);

	int withNext = 0;
	for (size_t k = 0; k < n; k++) {
		// 1. nextStates am Truncation-Schritt ist die Beobachtung VOR dem Reset
		bool isFinal = std::find(finalObsSeen.begin(), finalObsSeen.end(), diag.bootstrapStates[k]) != finalObsSeen.end();
		if (!isFinal)
			FAIL_AT("Truncation " << k << " (Position " << diag.envTruncIdx[k]
			        << "): Bootstrap-Zustand ist keine letzte Beobachtung einer Episode");

		// 2. GAE: Advantage = r + gamma * V(letzte Obs) - V(s); die Kette bricht dahinter ab
		float normRew = diag.rewards[k] / diag.returnStd;
		if (diag.rewardClipRange > 0)
			normRew = std::clamp(normRew, -diag.rewardClipRange, diag.rewardClipRange);
		CHECK_NEAR(diag.advantages[k], normRew + diag.gamma * diag.bootstrapValues[k] - diag.stateValues[k], 1e-5);

		if (!diag.envTruncHasNext[k]) continue;
		withNext++;
		// 3. ... und NICHT die Reset-Beobachtung der nächsten Episode (= states[t + 1])
		if (diag.bootstrapStates[k] == diag.followingStates[k])
			FAIL_AT("Truncation " << k << " (Position " << diag.envTruncIdx[k]
			        << "): bootstrappt von der Reset-Beobachtung (states[t+1])");
		// 4. Der Bootstrap-Wert ist nicht values[t + 1]
		CHECK_GT(std::abs(diag.bootstrapValues[k] - diag.followingValues[k]), 1e-7);
	}
	CHECK_GT(withNext, 4);
	// Diagnose-Metrik aus metrics.csv: Anteil der Timeouts mit Reset-Obs als Bootstrap-Zustand
	CHECK_NEAR(report["Trunc Bootstrap Reset Share"], 0.0, 1e-12);
	CHECK(report.data.count("Trunc Bootstrap V Diff") == 1);
}

TEST(K1b_Schalter_aus_Timeouts_sind_wieder_echte_Episodenenden) {
	if (!g_arenaReady) return;
	auto env = MakeNoTouchEnv(4, /*asTruncation*/ false);
	env.gym->Reset();
	Gym::StepResult result = {};
	for (int i = 0; i < 4; i++)
		result = env.gym->Step(IList(2, 0));
	CHECK(result.done);
	CHECK(!result.truncated);   // wie vor dem Audit: done ohne Truncation -> Ziel 0
	delete env.gym;
	delete env.match;
}

TEST(K1b_Schalter_kommt_ueber_EnvFactory_an) {
	if (!g_arenaReady) return;
	for (bool asTruncation : { true, false }) {
		TrainConfig cfg = {};
		cfg.numThreads = 1;
		cfg.numGamesPerThread = 1;
		cfg.noTouchTimeoutSecs = 4 * 8 / 120.f;   // 4 Schritte
		cfg.states = {};
		cfg.states.kickoff = 1;
		cfg.states.random = 0;
		cfg.timeoutsAsTruncation = asTruncation;
		EnvFactory factory(cfg);
		auto env = factory.Create();
		int truncations = 0;
		for (auto* cond : env.match->terminalConditions)
			truncations += cond->IsTruncation() ? 1 : 0;
		CHECK_EQ(truncations, asTruncation ? 2 : 0);   // NoTouch + Spielzeit, nie das Tor
		delete env.gym;
		delete env.match;
	}
}

#endif // RLGSC_HAS_FINAL_OBS

TEST(K1b_Upstream_Patch_hat_finalObs) {
	// Ein Upstream-Klon mit der ersten Patch-Fassung (Bootstrap von der Reset-Obs) darf nicht
	// unbemerkt gebaut werden; CMake prüft dasselbe (RLGSC_HAS_FINAL_OBS).
#ifdef RLGSC_HAS_FINAL_OBS
	CHECK(true);
#else
	FAIL_AT("third_party/patches/rlgympppo_cpp_truncation.patch ist in der alten Fassung angewendet (kein finalObs)");
#endif
}

TEST(Config_timeouts_as_truncation_Default_an_und_lesbar) {
	TrainConfig def = {};
	CHECK(def.timeoutsAsTruncation);

	auto path = std::filesystem::temp_directory_path() / "rlbot_test_cfg_trunc.json";
	std::ofstream(path) << R"({ "env": { "timeouts_as_truncation": false } })";
	auto cfg = TrainConfig::FromFile(path.string());
	std::filesystem::remove(path);
	CHECK(!cfg.timeoutsAsTruncation);

	// Roundtrip über config_used.json
	std::ofstream(path) << cfg.ToJSONString();
	auto again = TrainConfig::FromFile(path.string());
	std::filesystem::remove(path);
	CHECK(!again.timeoutsAsTruncation);
}
