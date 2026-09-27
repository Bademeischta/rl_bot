// Trainer: liest eine JSON-Config, baut die Environments und startet das PPO-Training.
//
//   train_bot.exe <config.json> [--collision-meshes <dir>] [--timestep-limit N] [--device cpu|cuda]
//                 [--extra-steps N] [--save-on-exit] [--stop-file <pfad>]
//
// --stop-file (Review-Befund R15): Sobald die Datei existiert, endet das Training nach der
// laufenden Iteration regulär; mit save_on_exit wird danach der End-Checkpoint geschrieben.
// So kann tools/experiments/run_experiment.ps1 einen Lauf abbrechen, ohne den Prozess mitten in
// einem Checkpoint-Save zu töten.
//
// Die verwendete Config wird beim Start in den Run-Ordner kopiert (config_used.json, mit
// Git-Hash des Builds und Startzeit), damit später nachvollziehbar ist, womit ein Checkpoint
// trainiert wurde (Audit H4).
#include "Config.h"
#include "EnvFactory.h"
#include "Metrics.h"

#include <RLGymPPO_CPP/Learner.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>

#ifndef RLBOT_GIT_HASH
#define RLBOT_GIT_HASH "unbekannt"
#endif

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
	RLbot::AccumRawReward(gameInst->match, gameMetrics);

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

static std::filesystem::path g_stopFile;
static bool g_stopRequested = false;

static void OnIteration(Learner* learner, Report& allMetrics) {
	RLbot::AggregateGameMetrics(learner->GetAllGameMetrics(), allMetrics);
	g_metricsCSV.Append(allMetrics);

	// Stop-Datei (R15): Die Learn()-Schleife prüft timestepLimit vor jeder Iteration, also endet
	// sie nach dieser. Ein fälliger periodischer Save dieser Iteration läuft vorher noch.
	std::error_code ec;
	if (!g_stopRequested && !g_stopFile.empty() && std::filesystem::exists(g_stopFile, ec)) {
		g_stopRequested = true;
		learner->config.timestepLimit = learner->totalTimesteps;
		RG_LOG("Stop-Datei " << g_stopFile << " gefunden: Training endet nach dieser Iteration bei "
			<< learner->totalTimesteps << " Steps");
	}
}

int main(int argc, char** argv) {
	if (argc < 2) {
		std::cerr << "usage: train_bot <config.json> [--collision-meshes <dir>] "
		             "[--timestep-limit N] [--device cpu|cuda] [--extra-steps N] [--save-on-exit] "
		             "[--stop-file <pfad>]\n";
		return 2;
	}

	std::string configPath = argv[1];
	std::string meshDir = "collision_meshes";
	int64_t timestepLimitOverride = -1;
	int64_t extraStepsOverride = -1;
	bool saveOnExitOverride = false;
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
		else if (arg == "--extra-steps") extraStepsOverride = std::stoll(next());
		else if (arg == "--save-on-exit") saveOnExitOverride = true;
		else if (arg == "--stop-file") g_stopFile = next();
		else { std::cerr << "Unbekanntes Argument: " << arg << "\n"; return 2; }
	}
	if (!g_stopFile.empty() && std::filesystem::exists(g_stopFile)) {
		// Eine alte Stop-Datei würde den Lauf nach der ersten Iteration beenden
		std::cerr << "Stop-Datei existiert schon: " << g_stopFile.string() << " (erst entfernen)\n";
		return 2;
	}

	RLbot::TrainConfig cfg = RLbot::TrainConfig::FromFile(configPath);
	if (timestepLimitOverride >= 0) cfg.timestepLimit = timestepLimitOverride;
	if (extraStepsOverride >= 0) cfg.extraSteps = extraStepsOverride;
	if (saveOnExitOverride) cfg.saveOnExit = true;
	if (!deviceOverride.empty()) cfg.device = deviceOverride;

	RG_LOG("Config: " << configPath);
	RG_LOG(cfg.ToJSONString());

	// Config neben die Checkpoints legen (Reproduzierbarkeit)
	std::filesystem::path runDir = std::filesystem::path(cfg.checkpointFolder).parent_path();
	if (!runDir.empty()) {
		std::filesystem::create_directories(runDir);
		// Audit H4: Git-Hash des Builds und Startzeit mitschreiben, damit Checkpoint und Code
		// verknüpft sind (-DRLBOT_GIT_HASH aus bench/cpp/build.ps1)
		std::time_t now = std::time(nullptr);
		char started[32];
		std::strftime(started, sizeof(started), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
		std::ofstream(runDir / "config_used.json") << cfg.ToUsedJSONString(RLBOT_GIT_HASH, started);
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

	// Step-Budget relativ zum geladenen Checkpoint (Experimente, Stufe 3)
	if (cfg.extraSteps > 0) {
		learner.config.timestepLimit = learner.totalTimesteps + cfg.extraSteps;
		RG_LOG("extra_steps: Lauf endet bei " << learner.config.timestepLimit
			<< " Steps (" << learner.totalTimesteps << " + " << cfg.extraSteps << ")");
	}

	learner.Learn();

	// End-Checkpoint, falls der letzte Save nicht auf dem Endstand liegt
	if (cfg.saveOnExit && !learner.config.checkpointSaveFolder.empty()) {
		auto endFolder = learner.config.checkpointSaveFolder / std::to_string(learner.totalTimesteps);
		if (!std::filesystem::exists(endFolder / "PPO_POLICY.lt")) {
			RG_LOG("save_on_exit: Checkpoint bei " << learner.totalTimesteps << " Steps schreiben");
			learner.Save();
		}
	}
	return 0;
}
