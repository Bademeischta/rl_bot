// Zeitbegrenzungen als Truncation (Audit K1).
//
// Harte Episodenlänge: RLGymSim_CPP bringt nur NoTouchCondition und GoalScoreCondition mit,
// eine reine Zeitbegrenzung fehlt. Ohne sie können Episoden bei ständigem Ballkontakt
// beliebig lang werden, was die Trajektorien-Verteilung verzerrt.
//
// Beide Zeitbegrenzungen (Spielzeit, NoTouch) beenden die Episode, ohne dass die Welt zu Ende
// ist. Mit dem Upstream-Patch third_party/patches/rlgympppo_cpp_truncation.patch melden sie
// das über IsTruncation(); der Learner bootstrappt dann den Wert des letzten Zustands statt
// ihn wie ein Tor auf 0 zu setzen (TorchFuncs::ComputeGAE). Ohne Patch ist IsTruncation()
// nur eine unbenutzte Methode und alles verhält sich wie vor dem Audit.
#pragma once

#include <RLGymSim_CPP/Utils/TerminalConditions/NoTouchCondition.h>
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
	virtual bool IsTruncation() const { return true; }
};

// NoTouchCondition des Upstreams, als Truncation markiert.
class NoTouchTruncation : public NoTouchCondition {
public:
	explicit NoTouchTruncation(int maxSteps) : NoTouchCondition(maxSteps) {}
	virtual bool IsTruncation() const { return true; }
};

} // namespace RLbot
