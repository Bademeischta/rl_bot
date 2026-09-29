// Geschwindigkeit (AUDIT.md §9): Optimierungen, die das Lernen nicht verändern dürfen, werden hier
// über den echten Pfad gegen das alte Verhalten auf identische Ergebnisse geprüft.
#include "test_util.h"

#include "train/cpp/Config.h"
#include "train/cpp/EnvFactory.h"

#include <RLGymPPO_CPP/Learner.h>
#include <RLGymPPO_CPP/Threading/ThreadAgentManager.h>
#include <RLGymPPO_CPP/PPO/PPOLearner.h>
#include <torch/cuda.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>

using namespace RLbot;

extern bool g_arenaReady;

// Wie die Hauptlauf-Config, dazu 2v2 (Spiele mit verschiedener Spielerzahl im selben Thread) und
// ein kurzer NoTouch-Timeout, damit viele Episoden per Truncation enden (finalObs-Pfad).
static TrainConfig SpeedTestConfig() {
	TrainConfig cfg = {};
	cfg.modeMix[0] = 1; cfg.modeMix[1] = 1; cfg.modeMix[2] = 0;
	cfg.noTouchTimeoutSecs = 1.5f;
	cfg.states.kickoffDrill = 4;
	cfg.rewards.goal = 5; cfg.rewards.concede = 5; cfg.rewards.teamSpirit = 0.1f;
	cfg.numThreads = 2;
	cfg.numGamesPerThread = 3;
	return cfg;
}

// Hält die Sammel-Threads an, bevor der Learner zerstört wird (auch wenn eine Prüfung wirft)
struct AgentStopper {
	RLGPC::ThreadAgentManager* mgr;
	~AgentStopper() { mgr->StopAgents(); }
};

static RLGPC::LearnerConfig SmallLearner(const TrainConfig& cfg) {
	RLGPC::LearnerConfig lc = MakeLearnerConfig(cfg);
	lc.timestepsPerIteration = 600;
	lc.expBufferSize = 1800;
	lc.ppo.batchSize = 600;
	lc.ppo.miniBatchSize = 300;
	lc.ppo.policyLayerSizes = { 32 };
	lc.ppo.criticLayerSizes = { 32 };
	lc.deviceType = RLGPC::LearnerDeviceType::CPU;
	lc.checkpointLoadFolder = "";
	lc.checkpointSaveFolder = "";
	lc.sendMetrics = false;
	lc.skillTrackerConfig.enabled = false;
	return lc;
}

// G2: Die Sammel-Threads schreiben die Schritte in einfache Arrays (RawTrajectory) statt je Spieler
// und Schritt ~20 Tensor-Operationen auszuführen. Mit verifyTrajectories füllen sie zusätzlich den
// alten Pfad; beide Ergebnisse müssen Bit für Bit gleich sein, in jedem Tensor und in der
// Truncation-Diagnose.
TEST(G2_Trajektorien_als_Arrays_sind_bitgleich_zum_alten_Tensor_Pfad) {
	if (!g_arenaReady) return;
	TrainConfig cfg = SpeedTestConfig();
	EnvFactory factory(cfg);
	RLGPC::LearnerConfig lc = SmallLearner(cfg);
	lc.verifyTrajectories = true;

	RLGPC::Learner learner([&]() { return factory.Create(); }, lc);
	auto* mgr = learner.agentMgr;
	mgr->StartAgents();
	AgentStopper stopper{ mgr };   // auch wenn eine Prüfung wirft: erst die Threads, dann der Learner
	int truncs = 0, rows = 0;
	for (int round = 0; round < 4; round++) {
		RLGPC::GameTrajectory now = mgr->CollectTimesteps(600);
		const RLGPC::GameTrajectory& old = mgr->lastLegacyResult;
		CHECK_EQ(now.size, old.size);
		CHECK(now.size >= 600);
		const char* names[] = { "states", "actions", "logProbs", "rewards", "nextStates", "dones", "truncateds" };
		for (size_t t = 0; t < RLGPC::TrajectoryTensors::TENSOR_AMOUNT; t++) {
			const torch::Tensor& a = now.data[t];
			torch::Tensor b = old.data[t].slice(0, 0, old.size);
			if (a.scalar_type() != b.scalar_type() || a.sizes() != b.sizes() || !torch::equal(a, b))
				FAIL_AT("Runde " << round << ": " << names[t] << " unterscheidet sich vom alten Pfad");
		}
		CHECK(mgr->lastEnvTruncIdx == mgr->lastLegacyEnvTruncIdx);
		CHECK(mgr->lastEnvTruncHasNext == mgr->lastLegacyEnvTruncHasNext);
		truncs += (int)mgr->lastEnvTruncIdx.size();
		rows += (int)now.data.states.size(0);
	}
	// Der Test deckt Episodenenden ab (Truncation per NoTouch/Drill, finalObs als nextState)
	CHECK_GT(truncs, 20);
	CHECK_GT(rows, 2400);
}

// G3: Ein Agent, der sein Sammel-Limit erreicht hat (maxCollect, z. B. mit collection_during_learn
// oder nach der letzten Iteration), wartete in einer Schleife ohne Blick auf shouldRun;
// StopAgents() hing dann für immer ("Stopping agents...", docs/phase0_results.md §3).
TEST(G3_StopAgents_haengt_nicht_am_Sammel_Limit) {
	if (!g_arenaReady) return;
	TrainConfig cfg = SpeedTestConfig();
	cfg.numThreads = 1;
	cfg.numGamesPerThread = 2;
	EnvFactory factory(cfg);
	RLGPC::LearnerConfig lc = SmallLearner(cfg);
	lc.timestepsPerIteration = 200;

	auto* learner = new RLGPC::Learner([&]() { return factory.Create(); }, lc);
	auto* mgr = learner->agentMgr;
	auto* agent = mgr->agents[0];
	mgr->StartAgents();
	// Niemand holt die Schritte ab: der Agent sammelt bis maxCollect und wartet dann
	auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
	while (agent->stepsCollected <= agent->maxCollect && std::chrono::steady_clock::now() < deadline)
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	CHECK(agent->stepsCollected > agent->maxCollect);

	// Eigener Thread: ohne den Fix kehrt StopAgents() nie zurück (der Learner bleibt dann bewusst stehen)
	auto stopped = std::make_shared<std::atomic<bool>>(false);
	std::thread([mgr, stopped]() { mgr->StopAgents(); *stopped = true; }).detach();
	deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
	while (!*stopped && std::chrono::steady_clock::now() < deadline)
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	if (!*stopped)
		FAIL_AT("StopAgents() haengt: der Agent wartet am Sammel-Limit ohne shouldRun zu pruefen");
	delete learner;
}

static std::vector<torch::Tensor> AllParams(RLGPC::Learner& l) {
	std::vector<torch::Tensor> out;
	for (auto& p : l.ppo->policy->parameters()) out.push_back(p.detach().cpu().clone());
	for (auto& p : l.ppo->valueNet->parameters()) out.push_back(p.detach().cpu().clone());
	return out;
}

static RLGPC::GameTrajectory CloneTrajectory(const RLGPC::GameTrajectory& t) {
	RLGPC::GameTrajectory c = t;
	for (size_t i = 0; i < RLGPC::TrajectoryTensors::TENSOR_AMOUNT; i++)
		c.data[i] = t.data[i].clone();
	return c;
}

// G4: Experience-Puffer im GPU-Speicher. Zwei Learner mit demselben Seed (Puffer auf der CPU wie
// bisher / auf der GPU) bekommen dieselben echt gesammelten Trajektorien und lernen darauf fünf
// Iterationen: Netze und Kennzahlen müssen bitgleich bleiben (gleiche Shuffle-Reihenfolge, gleiche
// Batches, nur ohne Host-zu-GPU-Kopie je Minibatch).
TEST(G4_Puffer_auf_der_GPU_lernt_bitgleich_zum_Puffer_auf_der_CPU) {
	if (!g_arenaReady || !torch::cuda::is_available()) return;
	TrainConfig cfg = SpeedTestConfig();
	EnvFactory factory(cfg);
	auto make = [&](bool onDevice) {
		RLGPC::LearnerConfig lc = SmallLearner(cfg);
		lc.deviceType = RLGPC::LearnerDeviceType::GPU_CUDA;
		lc.expBufferOnDevice = onDevice;
		lc.ppo.policyLayerSizes = { 64, 64 };
		lc.ppo.criticLayerSizes = { 64, 64 };
		lc.ppo.epochs = 2;
		return std::make_unique<RLGPC::Learner>([&]() { return factory.Create(); }, lc);
	};
	auto cpuBuf = make(false);
	auto gpuBuf = make(true);
	CHECK(!cpuBuf->expBuffer->storeOnDevice);
	CHECK(gpuBuf->expBuffer->storeOnDevice);

	auto sameParams = [&](const char* when) {
		auto pa = AllParams(*cpuBuf), pb = AllParams(*gpuBuf);
		CHECK_EQ(pa.size(), pb.size());
		for (size_t i = 0; i < pa.size(); i++)
			if (!torch::equal(pa[i], pb[i]))
				FAIL_AT(when << ": Parameter " << i << " unterscheidet sich");
	};
	sameParams("Start");

	auto* mgr = cpuBuf->agentMgr;
	mgr->StartAgents();
	AgentStopper stopper{ mgr };
	// 5 Iterationen: ab der dritten ist der Puffer (1800) voll, alte Daten werden verschoben
	for (int it = 0; it < 5; it++) {
		RLGPC::GameTrajectory traj = mgr->CollectTimesteps(600);
		mgr->disableCollection = true;   // wie im Learner: nicht sammeln, während gelernt wird
		RLGPC::GameTrajectory copy = CloneTrajectory(traj);
		RLGPC::Report ra, rb;
		cpuBuf->AddNewExperience(traj, ra);
		gpuBuf->AddNewExperience(copy, rb);
		CHECK(gpuBuf->expBuffer->data.states.is_cuda());
		CHECK(!cpuBuf->expBuffer->data.states.is_cuda());
		cpuBuf->ppo->Learn(cpuBuf->expBuffer, ra);
		gpuBuf->ppo->Learn(gpuBuf->expBuffer, rb);
		mgr->disableCollection = false;
		sameParams("nach Iteration");
		for (const char* key : { "Avg Advantage", "Avg Val Target", "Policy Entropy", "Mean KL Divergence",
		                         "Value Function Loss", "SB3 Clip Fraction", "Policy Update Magnitude" })
			CHECK_EQ(ra[key], rb[key]);
	}
}

static TrainConfig LoadSpeedConfig(const std::string& json) {
	static int counter = 0;
	auto path = std::filesystem::temp_directory_path() / ("rlbot_speed_cfg_" + std::to_string(counter++) + ".json");
	std::ofstream(path) << json;
	TrainConfig cfg = TrainConfig::FromFile(path.string());
	std::filesystem::remove(path);
	return cfg;
}

TEST(G4_Config_exp_buffer_on_device_Default_aus_und_kommt_im_Learner_an) {
	TrainConfig def = {};
	CHECK(!def.expBufferOnDevice);
	CHECK(!MakeLearnerConfig(def).expBufferOnDevice);
	TrainConfig cfg = LoadSpeedConfig(R"({"learner": {"exp_buffer_on_device": true}})");
	CHECK(cfg.expBufferOnDevice);
	CHECK(MakeLearnerConfig(cfg).expBufferOnDevice);
	// config_used.json trägt den Wert und ist wieder ladbar (R7)
	CHECK(LoadSpeedConfig(cfg.ToJSONString()).expBufferOnDevice);
}
