// Trainings-Config als JSON (Beispiel: train/configs/lucy_1v1.json).
// Alles, was ein Lauf braucht, steht in einer Datei, damit Läufe reproduzierbar sind.
#pragma once

#include "../../env/cpp/Rewards.h"
#include "../../env/cpp/StateSetters.h"

#include <RLGymPPO_CPP/LearnerConfig.h>

#include <string>
#include <vector>

namespace RLbot {

struct TrainConfig {
	// --- Environment ---
	int tickSkip = 8;
	int maxPlayers = 3;           // Padding-Obergrenze (3 = bis 3v3)
	int actionStackSize = 5;      // Lucy: k = 5
	float noTouchTimeoutSecs = 30.f;
	float gameTimeoutSecs = 300.f;
	// Anteile der Modi 1v1 / 2v2 / 3v3 an den Environments (werden normiert)
	float modeMix[3] = { 1.f, 0.f, 0.f };

	RewardWeights rewards = {};
	StateSetterWeights states = {};

	// --- Learner ---
	int numThreads = 16;
	int numGamesPerThread = 64;
	bool collectionDuringLearn = false;
	std::string device = "cuda";       // "cuda" | "cpu" | "auto"
	int64_t timestepLimit = 0;         // 0 = unbegrenzt
	int64_t timestepsPerIteration = 100000;
	int64_t timestepsPerSave = 25000000;   // Bauplan §2: 25 Mio.
	int checkpointsToKeep = 10;
	std::string checkpointFolder = "runs/default/checkpoints";

	std::vector<int> policyLayerSizes = { 512, 512, 512 };
	std::vector<int> criticLayerSizes = { 512, 512, 512 };
	int ppoEpochs = 2;
	int64_t ppoBatchSize = 100000;
	int64_t ppoMiniBatchSize = 50000;
	float entCoef = 0.01f;
	float clipRange = 0.2f;
	float policyLR = 2e-4f;
	float criticLR = 2e-4f;
	float gaeLambda = 0.95f;
	float gaeGamma = 0.9954f;          // ~10 s Horizont bei TickSkip 8
	int randomSeed = 123;

	// --- Metriken / Skill ---
	bool sendMetrics = false;          // true braucht wandb im Python-Env
	std::string metricsProjectName = "rlbot";
	std::string metricsGroupName = "runs";
	std::string metricsRunName = "lucy";
	bool skillTrackerEnabled = true;
	int64_t skillTimestepsPerVersion = 500000000;
	int skillMaxVersions = 20;         // PFSP-Pool (Bauplan: 15-20)
	// Alle wie viele Iterationen Eval-Spiele laufen. Upstream-Standard 4 kostet rund 8 %
	// Durchsatz; 16 senkt das auf etwa 2 % und liefert immer noch regelmäßig ein Rating.
	int skillUpdateInterval = 16;

	// Lädt eine JSON-Datei; unbekannte Felder sind ein Fehler (Tippfehler sollen auffallen).
	static TrainConfig FromFile(const std::string& path);
	std::string ToJSONString() const;
};

// Überträgt die Learner-Felder in die RLGymPPO-Config.
RLGPC::LearnerConfig MakeLearnerConfig(const TrainConfig& cfg);

} // namespace RLbot
