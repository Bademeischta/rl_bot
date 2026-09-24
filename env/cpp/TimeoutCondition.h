// Harte Episodenlänge. RLGymSim_CPP bringt nur NoTouchCondition und GoalScoreCondition mit,
// eine reine Zeitbegrenzung fehlt. Ohne sie können Episoden bei ständigem Ballkontakt
// beliebig lang werden, was die Trajektorien-Verteilung verzerrt.
#pragma once

#include <RLGymSim_CPP/Utils/TerminalConditions/TerminalCondition.h>

namespace RLbot {
using namespace RLGSC;

class TimeoutCondition : public TerminalCondition {
public:
	int steps = 0;
	int maxSteps;

	explicit TimeoutCondition(int maxSteps) : maxSteps(maxSteps) {}

	virtual void Reset(const GameState& initialState) { steps = 0; }
	virtual bool IsTerminal(const GameState& currentState) { return ++steps >= maxSteps; }
};

} // namespace RLbot
