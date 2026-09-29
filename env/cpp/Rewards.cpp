#include "Rewards.h"

#include <RLGymSim_CPP/Math.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/ZeroSumReward.h>

#include <algorithm>

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

void KickoffFirstTouchReward::Reset(const GameState& s) {
	// Wie PlayTracker: Anstoß = ruhender Ball in der Mitte
	armed = std::abs(s.ball.pos.x) < 1 && std::abs(s.ball.pos.y) < 1 && s.ball.pos.z < 100 && s.ball.vel.Length() < 1;
	resetTick = s.lastTickCount;
	winners.clear();
}

void KickoffFirstTouchReward::PreStep(const GameState& s) {
	winners.clear();
	if (!armed)
		return;
	uint64_t first = ~0ULL;
	for (auto& p : s.players) {
		const auto& hit = p.carState.ballHitInfo;
		if (p.ballTouchedStep && hit.isValid && hit.tickCountWhenHit >= resetTick)
			first = std::min(first, hit.tickCountWhenHit);
	}
	if (first == ~0ULL)
		return;
	for (auto& p : s.players)
		if (p.ballTouchedStep && p.carState.ballHitInfo.isValid && p.carState.ballHitInfo.tickCountWhenHit == first)
			winners.push_back(p.carId);
	armed = false;
}

float KickoffFirstTouchReward::GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
	return std::find(winners.begin(), winners.end(), player.carId) != winners.end() ? 1.f : 0.f;
}

void PotentialReward::Reset(const GameState& s) {
	phi->Reset(s);
	last.clear();
	for (auto& p : s.players)
		last[p.carId] = phi->GetReward(p, s, Action());
}

float PotentialReward::Step(const PlayerData& player, float next) {
	auto it = last.find(player.carId);
	float prev = it != last.end() ? it->second : next;
	last[player.carId] = next;
	return gamma * next - prev;
}

float PotentialReward::GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
	return Step(player, phi->GetReward(player, state, prevAction));
}

float PotentialReward::GetFinalReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
	// Tor = echtes Ende, Potenzial 0; sonst (Truncation) läuft die Welt weiter
	bool terminal = RLGSC::Math::IsBallScored(state.ball.pos);
	return Step(player, terminal ? 0.f : phi->GetReward(player, state, prevAction));
}

float AirTouchReward::GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
	const auto& hit = player.carState.ballHitInfo;
	if (!player.ballTouchedStep || !hit.isValid || player.carState.isOnGround)
		return 0;
	auto it = lastRewardTick.find(player.carId);
	if (it != lastRewardTick.end() && hit.tickCountWhenHit < it->second + COOLDOWN_TICKS)
		return 0;
	lastRewardTick[player.carId] = hit.tickCountWhenHit;
	float h = (hit.ballPos.z - CommonValues::BALL_RADIUS) / (CommonValues::CEILING_Z - CommonValues::BALL_RADIUS);
	return RS_CLAMP(h, 0.f, 1.f);
}

std::vector<RewardPart> BuildLucyRewardParts(const RewardWeights& w) {
	std::vector<RewardPart> parts;

	// Ereignis-Rewards: Tor, Gegentor, Demo. teamGoal deckt auch das eigene Tor ab,
	// deshalb kein separates "goal", sonst zählt ein eigenes Tor doppelt.
	EventReward::WeightScales scales = {};
	scales.teamGoal = w.goal;
	scales.concede = -w.concede;
	scales.demo = w.demo;
	scales.demoed = -w.demoed;
	if (w.goal != 0 || w.concede != 0 || w.demo != 0 || w.demoed != 0)
		parts.push_back({ "event", new EventReward(scales), 1.f });

	if (w.touchBallToGoalAccel != 0)
		parts.push_back({ "touch_ball_to_goal_accel", new TouchBallToGoalAccelReward(), w.touchBallToGoalAccel });
	// potential_shaping_scale > 0: dieselben KRC-Terme als Potenzialdifferenz (Spieltest, Ecken-Schleife)
	float k = w.potentialShapingScale;
	auto shaping = [&](RewardFunction* fn) -> RewardFunction* {
		return k > 0 ? new PotentialReward(fn, w.potentialGamma) : fn;
	};
	const char* suffix = k > 0 ? "_pot" : "";
	if (w.offensivePotential != 0)
		parts.push_back({ std::string("offensive_potential_krc") + suffix, shaping(MakeOffensivePotential()),
		                  w.offensivePotential * (k > 0 ? k : 1.f) });
	if (w.distWeightedAlign != 0)
		parts.push_back({ std::string("dist_weighted_align_krc") + suffix, shaping(MakeDistWeightedAlignment()),
		                  w.distWeightedAlign * (k > 0 ? k : 1.f) });
	if (w.velocityPlayerToBall != 0)
		parts.push_back({ "velocity_player_to_ball", new VelocityPlayerToBallReward(), w.velocityPlayerToBall });
	if (w.saveBoost != 0)
		parts.push_back({ "save_boost", new SaveBoostReward(0.5f), w.saveBoost });
	if (w.inAir != 0)
		parts.push_back({ "in_air", new InAirReward(), w.inAir });
	if (w.kickoffFirstTouch != 0)
		parts.push_back({ "kickoff_first_touch", new KickoffFirstTouchReward(), w.kickoffFirstTouch });
	if (w.airTouch != 0)
		parts.push_back({ "air_touch", new AirTouchReward(), w.airTouch });
	return parts;
}

RewardFunction* BuildLucyReward(const RewardWeights& w) {
	std::vector<std::pair<RewardFunction*, float>> parts;
	for (auto& part : BuildLucyRewardParts(w))
		parts.push_back({ part.fn, part.weight });

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
