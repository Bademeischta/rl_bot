// Erzeugt die Trainings-Environments aus einer TrainConfig.
//
// Teamgrößen: RLGymSim kann die Spielerzahl innerhalb eines Matches nicht ändern
// ("Changing number of players at state reset is currently not supported"), deshalb wird
// der Modus-Mix über die Environments verteilt: Jedes Env bekommt beim Erzeugen eine feste
// Teamgröße, die Anteile ergeben sich aus env.mode_mix (Bauplan §4, ein Padding-Modell für alle Modi).
#pragma once

#include "Config.h"

#include <RLGymPPO_CPP/Threading/GameInst.h>

#include <atomic>

namespace RLbot {

class EnvFactory {
public:
	TrainConfig cfg;

	// Zähler über alle erzeugten Envs, daraus werden Teamgröße und Seed deterministisch
	// abgeleitet (Envs werden im Hauptthread nacheinander erzeugt, die Reihenfolge ist fix).
	std::atomic<int> envCounter = 0;
	std::atomic<int> evalCounter = 0;

	// Vorberechnete Modus-Zuteilung; TeamSizeForIndex greift nur noch darauf zu und ist
	// damit thread-sicher (Envs werden aus mehreren Threads erzeugt).
	constexpr static int SCHEDULE_SIZE = 4096;
	std::vector<int> schedule;

	explicit EnvFactory(const TrainConfig& cfg);

	// Teamgröße für das n-te Environment nach dem Modus-Mix.
	int TeamSizeForIndex(int index) const;

	RLGPC::EnvCreateResult Create();

	// Env-Erzeuger nur für Evaluation: immer Kickoff-Start, feste Teamgröße.
	RLGPC::EnvCreateResult CreateEval(int teamSize);
};

} // namespace RLbot
