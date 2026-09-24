// Phase-0 C++-SPS-Benchmark auf RLGymPPO_CPP (gleiches Setup wie bench/python_sps.py:
// 1v1, TickSkip 8, DefaultOBS, DiscreteAction(90), Netz 3x256, 100k Steps/Iteration, 1 Epoche).
//
// Aufruf: bench_cpp_sps.exe <cpu|cuda> <numThreads> <gamesPerThread> <steps> <csvOut> <meshDir> [collectDuringLearn 0|1]
#include <RLGymPPO_CPP/Learner.h>

#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>
#include <RLGymSim_CPP/Utils/OBSBuilders/DefaultOBS.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/StateSetters/KickoffState.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/NoTouchCondition.h>

#include <fstream>
#include <iostream>
#include <string>

using namespace RLGPC;
using namespace RLGSC;

static std::ofstream g_csv;
static std::string g_device;
static int g_threads = 0, g_games = 0, g_itr = 0;

EnvCreateResult EnvCreateFunc() {
	constexpr int TICK_SKIP = 8;
	auto rewards = new CombinedReward({
		{ new EventReward({ .teamGoal = 1.f, .concede = -1.f, .touch = 0.01f }), 10.f },
	});
	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(30 * 120 / TICK_SKIP),
		new GoalScoreCondition(),
	};
	Match* match = new Match(rewards, terminalConditions, new DefaultOBS(), new DiscreteAction(),
		new KickoffState(), 1, true);
	return { match, new Gym(match, TICK_SKIP) };
}

void OnIteration(Learner* learner, Report& r) {
	g_itr++;
	g_csv << g_device << ',' << g_threads << ',' << g_games << ',' << g_itr << ','
		<< (learner->config.collectionDuringLearn ? 1 : 0) << ','
		<< r["Collected Steps/Second"] << ',' << r["Overall Steps/Second"] << ','
		<< r["Collection Time"] << ',' << r["Consumption Time"] << ','
		<< r["Cumulative Timesteps"] << '\n';
	g_csv.flush();
}

int main(int argc, char** argv) {
	if (argc < 7) {
		std::cerr << "usage: bench_cpp_sps <cpu|cuda> <numThreads> <gamesPerThread> <steps> <csvOut> <meshDir>\n";
		return 2;
	}
	g_device = argv[1];
	g_threads = std::stoi(argv[2]);
	g_games = std::stoi(argv[3]);
	uint64_t steps = std::stoull(argv[4]);

	g_csv.open(argv[5], std::ios::app);
	RocketSim::Init(argv[6]);

	LearnerConfig cfg = {};
	cfg.deviceType = g_device == "cuda" ? LearnerDeviceType::GPU_CUDA : LearnerDeviceType::CPU;
	cfg.numThreads = g_threads;
	cfg.numGamesPerThread = g_games;
	cfg.timestepLimit = steps;
	cfg.collectionDuringLearn = argc > 7 && std::stoi(argv[7]) != 0;

	int tsPerItr = 100 * 1000;
	cfg.timestepsPerIteration = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = 50 * 1000;
	cfg.expBufferSize = tsPerItr * 3;
	cfg.ppo.epochs = 1;
	cfg.ppo.entCoef = 0.01f;
	cfg.ppo.policyLR = 2e-4;
	cfg.ppo.criticLR = 2e-4;
	cfg.ppo.policyLayerSizes = { 256, 256, 256 };
	cfg.ppo.criticLayerSizes = { 256, 256, 256 };

	cfg.sendMetrics = false;
	cfg.renderMode = false;
	cfg.checkpointLoadFolder = "";
	cfg.checkpointSaveFolder = "";

	Learner learner(EnvCreateFunc, cfg);
	learner.iterationCallback = OnIteration;
	learner.Learn();
	return 0;
}
