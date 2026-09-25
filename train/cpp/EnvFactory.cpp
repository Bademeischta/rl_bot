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
		new NoTouchTruncation((int)(cfg.noTouchTimeoutSecs * 120 / ticksPerStep)),
		new TimeoutCondition((int)(cfg.gameTimeoutSecs * 120 / ticksPerStep)),
		new GoalScoreCondition(),
	};
}

// Seed-Ströme je Environment: 0 = State-Setter, 1 = Obs-Shuffle (Audit H6).
// Eval-Envs bekommen einen eigenen Indexbereich, damit sie die Trainings-Envs nicht verschieben.
constexpr int STREAM_STATE = 0, STREAM_OBS = 1;
constexpr int EVAL_INDEX_BASE = 1 << 20;

static StackedPaddedOBS* MakeObs(const TrainConfig& cfg, int envIndex) {
	int64_t seed = cfg.seedEnvs ? (int64_t)SeedForEnv(cfg.randomSeed, envIndex, STREAM_OBS) : -1;
	StackedPaddedOBS defaults;   // nur für die Normierungskoeffizienten
	return new StackedPaddedOBS(cfg.maxPlayers, cfg.actionStackSize, cfg.shuffleSlots,
	                            defaults.posCoef, defaults.velCoef, defaults.angVelCoef,
	                            defaults.padTimerCoef, seed);
}

RLGPC::EnvCreateResult EnvFactory::Create() {
	int index = envCounter++;
	int teamSize = TeamSizeForIndex(index);

	auto* setter = cfg.seedEnvs
		? new WeightedStateSetter(cfg.states, SeedForEnv(cfg.randomSeed, index, STREAM_STATE))
		: new WeightedStateSetter(cfg.states);

	auto* match = new Match(
		BuildLucyReward(cfg.rewards),
		MakeTerminalConditions(cfg),
		MakeObs(cfg, index),
		new DiscreteAction(),
		setter,
		teamSize,
		true
	);
	return { match, new Gym(match, cfg.tickSkip) };
}

RLGPC::EnvCreateResult EnvFactory::CreateEval(int teamSize) {
	int index = EVAL_INDEX_BASE + evalCounter++;
	auto* match = new Match(
		BuildLucyReward(cfg.rewards),
		MakeTerminalConditions(cfg),
		MakeObs(cfg, index),
		new DiscreteAction(),
		new KickoffSetter(),
		teamSize,
		true
	);
	return { match, new Gym(match, cfg.tickSkip) };
}

} // namespace RLbot
