// Trainer: liest eine JSON-Config, baut die Environments und startet das PPO-Training.
//
//   train_bot.exe <config.json> [--collision-meshes <dir>] [--timestep-limit N] [--device cpu|cuda]
//
// Die verwendete Config wird beim Start in den Run-Ordner kopiert, damit später nachvollziehbar
// ist, womit ein Checkpoint trainiert wurde.
#include "Config.h"
#include "EnvFactory.h"
#include "Metrics.h"

#include <RLGymPPO_CPP/Learner.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace RLGPC;
using namespace RLGSC;

static RLbot::EnvFactory* g_factory = nullptr;

// DisplayReport() des Upstreams druckt nur eine feste Whitelist, eigene Metriken tauchen
// dort nie auf. Ohne wandb wären sie damit verloren, deshalb schreiben wir jede Iteration
// zusätzlich in eine CSV (Audit M1: Kopfzeile aus der Datei übernehmen, neue Spalten anhängen).
static RLbot::MetricsCSVWriter g_metricsCSV;

static RLbot::EpisodeLengthTracker g_episodeLengths;

// Wird aus vielen Threads gleichzeitig aufgerufen: nur die Argumente und den
// (intern gesperrten) Längen-Tracker anfassen.
static void OnStep(GameInst* gameInst, const Gym::StepResult& stepResult, Report& gameMetrics) {
	// Skill-Eval-Spiele haben eigene Reports, die nie ausgelesen werden: nicht mitzählen.
	if (gameInst->isEval)
		return;

	RLbot::AccumStepMetrics(stepResult.state, gameMetrics);

	if (stepResult.done) {
		// GameInst::Step erhöht totalSteps erst nach dem Callback, der aktuelle Step zählt also mit.
		uint64_t length = g_episodeLengths.OnEpisodeEnd(gameInst, gameInst->totalSteps + 1);
		bool truncated = false;
#ifdef RLGSC_HAS_TRUNCATION
		truncated = stepResult.truncated;
#endif
		auto end = RLbot::ClassifyEpisodeEnd(stepResult.state, gameInst->match, truncated);
		RLbot::AccumEpisodeEnd(end, length, RLbot::CurrentSceneName(gameInst->match), gameMetrics);
	}
}

static void OnIteration(Learner* learner, Report& allMetrics) {
	RLbot::AggregateGameMetrics(learner->GetAllGameMetrics(), allMetrics);
	g_metricsCSV.Append(allMetrics);
}

int main(int argc, char** argv) {
	if (argc < 2) {
		std::cerr << "usage: train_bot <config.json> [--collision-meshes <dir>] "
		             "[--timestep-limit N] [--device cpu|cuda]\n";
		return 2;
	}

	std::string configPath = argv[1];
	std::string meshDir = "collision_meshes";
	int64_t timestepLimitOverride = -1;
	std::string deviceOverride;

	for (int i = 2; i < argc; i++) {
		std::string arg = argv[i];
		auto next = [&]() -> std::string {
			if (i + 1 >= argc) { std::cerr << "Fehlender Wert für " << arg << "\n"; exit(2); }
			return argv[++i];
		};
		if (arg == "--collision-meshes") meshDir = next();
		else if (arg == "--timestep-limit") timestepLimitOverride = std::stoll(next());
		else if (arg == "--device") deviceOverride = next();
		else { std::cerr << "Unbekanntes Argument: " << arg << "\n"; return 2; }
	}

	RLbot::TrainConfig cfg = RLbot::TrainConfig::FromFile(configPath);
	if (timestepLimitOverride >= 0) cfg.timestepLimit = timestepLimitOverride;
	if (!deviceOverride.empty()) cfg.device = deviceOverride;

	RG_LOG("Config: " << configPath);
	RG_LOG(cfg.ToJSONString());

	// Config neben die Checkpoints legen (Reproduzierbarkeit)
	std::filesystem::path runDir = std::filesystem::path(cfg.checkpointFolder).parent_path();
	if (!runDir.empty()) {
		std::filesystem::create_directories(runDir);
		std::ofstream(runDir / "config_used.json") << cfg.ToJSONString();
		g_metricsCSV = RLbot::MetricsCSVWriter(runDir / "metrics.csv");
	}

	RocketSim::Init(meshDir);

	RLbot::EnvFactory factory(cfg);
	g_factory = &factory;

	LearnerConfig lc = RLbot::MakeLearnerConfig(cfg);
	if (cfg.skillTrackerEnabled) {
		// Skill-Eval immer im kleinsten aktiven Modus, damit die Zahlen vergleichbar bleiben
		int evalTeamSize = cfg.modeMix[0] > 0 ? 1 : (cfg.modeMix[1] > 0 ? 2 : 3);
		lc.skillTrackerConfig.envCreateFunc = [evalTeamSize]() {
			return g_factory->CreateEval(evalTeamSize);
		};
	}

	Learner learner([]() { return g_factory->Create(); }, lc);
	learner.stepCallback = OnStep;
	learner.iterationCallback = OnIteration;
	learner.Learn();
	return 0;
}
