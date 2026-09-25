// Zeitbegrenzungen als Truncation (Audit K1).
//
// Harte Episodenlänge: RLGymSim_CPP bringt nur NoTouchCondition und GoalScoreCondition mit,
// eine reine Zeitbegrenzung fehlt. Ohne sie können Episoden bei ständigem Ballkontakt
// beliebig lang werden, was die Trajektorien-Verteilung verzerrt.
//
// Beide Zeitbegrenzungen (Spielzeit, NoTouch) beenden die Episode, ohne dass die Welt zu Ende
// ist. Mit dem Upstream-Patch third_party/patches/rlgympppo_cpp_truncation.patch melden sie
// das über IsTruncation(); der Learner bootstrappt dann den Wert der letzten Beobachtung der
// Episode statt ihn wie ein Tor auf 0 zu setzen (TorchFuncs::ComputeGAE).
//
// Schalter env.timeouts_as_truncation (Review-Befund R4, Default true): false meldet die
// Zeitbegrenzungen wieder als echtes Episodenende (Ziel 0 wie vor dem Audit), für den Rückweg
// und für einen A/B-Vergleich. Ohne Patch ist IsTruncation() eine unbenutzte Methode.
#pragma once

#include <RLGymSim_CPP/Utils/TerminalConditions/NoTouchCondition.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/TerminalCondition.h>

namespace RLbot {
using namespace RLGSC;

class TimeoutCondition : public TerminalCondition {
public:
	int steps = 0;
	int maxSteps;
	bool asTruncation;

	explicit TimeoutCondition(int maxSteps, bool asTruncation = true)
		: maxSteps(maxSteps), asTruncation(asTruncation) {}

	virtual void Reset(const GameState& initialState) { steps = 0; }
	virtual bool IsTerminal(const GameState& currentState) { return ++steps >= maxSteps; }
	virtual bool IsTruncation() const { return asTruncation; }
};

// NoTouchCondition des Upstreams, als Truncation markiert (abschaltbar wie oben).
class NoTouchTruncation : public NoTouchCondition {
public:
	bool asTruncation;

	explicit NoTouchTruncation(int maxSteps, bool asTruncation = true)
		: NoTouchCondition(maxSteps), asTruncation(asTruncation) {}
	virtual bool IsTruncation() const { return asTruncation; }
};

} // namespace RLbot
