#include "EnvFactory.h"

#include "../../env/cpp/Obs.h"
#include "../../env/cpp/TimeoutCondition.h"

#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/NoTouchCondition.h>

namespace RLbot {

EnvFactory::EnvFactory(const TrainConfig& cfg) : cfg(cfg) {
	// Größte-Reste-Verfahren: Jeder Platz geht an den Modus, der seinem Sollanteil am
	// weitesten hinterherhinkt. Das trifft die Gewichte exakt, ist deterministisch und
	// verteilt die Modi gleichmäßig statt in Blöcken (wichtig, weil die Env-Anzahl
	// klein sein kann und alle Modi von Anfang an vertreten sein sollen).
	double share[3];
	double sum = cfg.modeMix[0] + cfg.modeMix[1] + cfg.modeMix[2];
	for (int i = 0; i < 3; i++)
		share[i] = cfg.modeMix[i] / sum;

	double assigned[3] = { 0, 0, 0 };
	schedule.resize(SCHEDULE_SIZE);
	for (int slot = 0; slot < SCHEDULE_SIZE; slot++) {
		int best = -1;
		double bestDeficit = 0;
		for (int i = 0; i < 3; i++) {
			if (share[i] <= 0)
				continue;
			double deficit = share[i] * (slot + 1) - assigned[i];
			if (best < 0 || deficit > bestDeficit) {
				best = i;
				bestDeficit = deficit;
			}
		}
		schedule[slot] = best + 1;
		assigned[best] += 1;
	}
}

int EnvFactory::TeamSizeForIndex(int index) const {
	return schedule[((index % SCHEDULE_SIZE) + SCHEDULE_SIZE) % SCHEDULE_SIZE];
}

static std::vector<TerminalCondition*> MakeTerminalConditions(const TrainConfig& cfg) {
	int ticksPerStep = cfg.tickSkip;
	return {
		new NoTouchCondition((int)(cfg.noTouchTimeoutSecs * 120 / ticksPerStep)),
		new TimeoutCondition((int)(cfg.gameTimeoutSecs * 120 / ticksPerStep)),
		new GoalScoreCondition(),
	};
}

RLGPC::EnvCreateResult EnvFactory::Create() {
	int teamSize = TeamSizeForIndex(envCounter++);

	auto* match = new Match(
		BuildLucyReward(cfg.rewards),
		MakeTerminalConditions(cfg),
		new StackedPaddedOBS(cfg.maxPlayers, cfg.actionStackSize, true),
		new DiscreteAction(),
		new WeightedStateSetter(cfg.states),
		teamSize,
		true
	);
	return { match, new Gym(match, cfg.tickSkip) };
}

RLGPC::EnvCreateResult EnvFactory::CreateEval(int teamSize) {
	auto* match = new Match(
		BuildLucyReward(cfg.rewards),
		MakeTerminalConditions(cfg),
		new StackedPaddedOBS(cfg.maxPlayers, cfg.actionStackSize, true),
		new DiscreteAction(),
		new KickoffSetter(),
		teamSize,
		true
	);
	return { match, new Gym(match, cfg.tickSkip) };
}

} // namespace RLbot
