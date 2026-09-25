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
	// Mitspieler-/Gegner-Slots im Obs bei jedem Schritt mischen (Audit H3). true = bisheriges
	// Verhalten; false ist als Experiment gedacht (train/configs/experiments/h3_no_shuffle.json).
	bool shuffleSlots = true;
	// Eigene State-Setter und den Slot-Shuffle aus learner.random_seed + Env-Index seeden
	// (Audit H6). false = RocketSims zeitgeseedeter Engine wie vor dem Audit. Default false
	// (Review-Befund R6): bestehende Configs wie lucy_1v1.json verhalten sich unverändert; die
	// Experiment-Configs (train/configs/experiments/) setzen true ausdrücklich.
	bool seedEnvs = false;
	// NoTouch- und Spielzeit-Timeout als Truncation melden (Audit K1, Review-Befund R4): der
	// Learner bootstrappt dann vom Wert der letzten Beobachtung der Episode. false = Timeouts
	// sind wieder echte Episodenenden mit Ziel 0 wie vor dem Audit (Rückweg, A/B-Vergleich).
	bool timeoutsAsTruncation = true;

	RewardWeights rewards = {};
	StateSetterWeights states = {};

	// --- Learner ---
	int numThreads = 16;
	int numGamesPerThread = 64;
	bool collectionDuringLearn = false;
	std::string device = "cuda";       // "cuda" | "cpu" | "auto"
	int64_t timestepLimit = 0;         // 0 = unbegrenzt (absolute Step-Zahl)
	// Step-Budget RELATIV zum geladenen Checkpoint (für Experimente, Stufe 3): Lauf endet bei
	// Checkpoint-Steps + extra_steps. 0 = aus (nur timestep_limit zählt, bisheriges Verhalten).
	int64_t extraSteps = 0;
	// Am Ende eines begrenzten Laufs noch einen Checkpoint schreiben, falls der letzte Save
	// nicht genau auf dem Endstand liegt. false = bisheriges Verhalten (kein End-Save).
	bool saveOnExit = false;
	int64_t timestepsPerIteration = 100000;
	int64_t timestepsPerSave = 25000000;   // Bauplan §2: 25 Mio.
	int checkpointsToKeep = 10;
	std::string checkpointFolder = "runs/default/checkpoints";

	std::vector<int> policyLayerSizes = { 512, 512, 512 };
	std::vector<int> criticLayerSizes = { 512, 512, 512 };
	int ppoEpochs = 2;
	// Größe des Experience-Puffers in Iterationen (Audit H5): Puffer = timesteps_per_iteration
	// mal dieser Faktor. Mit ppo_batch_size = timesteps_per_iteration ergibt das
	// exp_buffer_iterations Batches pro Epoche, also ppo_epochs * exp_buffer_iterations
	// Gradientenschritte pro Iteration (Default 3 * 2 = 6, wie bisher hartkodiert).
	int expBufferIterations = 3;
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
	// Felder mit führendem "_" (_comment, _git, _started) werden auf jeder Ebene ignoriert.
	static TrainConfig FromFile(const std::string& path);
	std::string ToJSONString() const;
	// Inhalt von config_used.json (Audit H4): ToJSONString() plus _git und _started. Mit
	// FromFile wieder ladbar (Review-Befund R7), z. B. um einen Lauf exakt zu wiederholen.
	std::string ToUsedJSONString(const std::string& gitHash, const std::string& started) const;
};

// Überträgt die Learner-Felder in die RLGymPPO-Config.
RLGPC::LearnerConfig MakeLearnerConfig(const TrainConfig& cfg);

} // namespace RLbot
