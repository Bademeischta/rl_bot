#include "Rewards.h"

#include <RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/ZeroSumReward.h>

namespace RLbot {

// Skalen für die Distanz-Rewards. Die exakten Lucy-Konstanten sind nicht veröffentlicht;
// hier: Spieler-Ball auf Auto-Höchstgeschwindigkeit, Ball-Tor auf Ball-Höchstgeschwindigkeit
// normiert. Beide sind bewusst Parameter, damit sie ablatierbar bleiben.
static constexpr float DIST_SCALE_PLAYER_BALL = CommonValues::CAR_MAX_SPEED;
static constexpr float DIST_SCALE_BALL_GOAL = CommonValues::BALL_MAX_SPEED;

RewardFunction* MakeOffensivePotential() {
	return new KRCReward({
		new AlignBallGoalReward(1.f, 0.f),
		new VelocityPlayerToBallReward(),
		new ParamDistanceReward(ParamDistanceReward::Target::PLAYER_TO_BALL, DIST_SCALE_PLAYER_BALL),
	});
}

RewardFunction* MakeDistWeightedAlignment() {
	return new KRCReward({
		new AlignBallGoalReward(1.f, 0.f),
		new ParamDistanceReward(ParamDistanceReward::Target::PLAYER_TO_BALL, DIST_SCALE_PLAYER_BALL),
	});
}

RewardFunction* BuildLucyReward(const RewardWeights& w) {
	std::vector<std::pair<RewardFunction*, float>> parts;

	// Ereignis-Rewards: Tor, Gegentor, Demo. teamGoal deckt auch das eigene Tor ab,
	// deshalb kein separates "goal", sonst zählt ein eigenes Tor doppelt.
	EventReward::WeightScales scales = {};
	scales.teamGoal = w.goal;
	scales.concede = -w.concede;
	scales.demo = w.demo;
	scales.demoed = -w.demoed;
	if (w.goal != 0 || w.concede != 0 || w.demo != 0 || w.demoed != 0)
		parts.push_back({ new EventReward(scales), 1.f });

	if (w.touchBallToGoalAccel != 0)
		parts.push_back({ new TouchBallToGoalAccelReward(), w.touchBallToGoalAccel });
	if (w.offensivePotential != 0)
		parts.push_back({ MakeOffensivePotential(), w.offensivePotential });
	if (w.distWeightedAlign != 0)
		parts.push_back({ MakeDistWeightedAlignment(), w.distWeightedAlign });
	if (w.velocityPlayerToBall != 0)
		parts.push_back({ new VelocityPlayerToBallReward(), w.velocityPlayerToBall });
	if (w.saveBoost != 0)
		parts.push_back({ new SaveBoostReward(0.5f), w.saveBoost });
	if (w.inAir != 0)
		parts.push_back({ new InAirReward(), w.inAir });

	// Der Tap merkt sich die Rewards vor einem Zero-Sum-Wrapper (raw_step_reward, Review R10)
	RewardFunction* combined = new RawRewardTap(new CombinedReward(parts, true));

	// teamSpirit > 0 -> zero-sum und Teamverteilung (Bauplan: tau startet bei 0 und
	// steigt gate-getriggert). Bei 0 bleibt es beim reinen Eigenreward.
	// Im 1v1 ist tau wirkungslos: Das eigene Team ist nur der Spieler selbst, also
	// r_i' = r_i * (1 - tau) + r_i * tau - r_j = r_i - r_j für jedes tau > 0 (Review R10).
	// Tor/Gegentor zählen dabei doppelt (+g beim Schützen, -g beim Gegner wird abgezogen).
	if (w.teamSpirit > 0)
		return new ZeroSumReward(combined, w.teamSpirit, 1.f, true);
	return combined;
}

const RawRewardTap* FindRawRewardTap(const RewardFunction* fn) {
	if (auto* tap = dynamic_cast<const RawRewardTap*>(fn))
		return tap;
	if (auto* zeroSum = dynamic_cast<const ZeroSumReward*>(fn))
		return dynamic_cast<const RawRewardTap*>(zeroSum->childFunc);
	return nullptr;
}

} // namespace RLbot
