// Lucy-SKG-nahe Reward-Funktionen (Bauplan v2 §6).
//
// Kern ist die KRC (Kinesthetic Reward Combination, arXiv 2305.15801 eq. 2):
//   R_c = sgn * (prod |R_i|)^(1/n),  sgn = +1 nur wenn alle Komponenten > 0, sonst -1
// und die parametrisierte Distanz-Belohnung (eq. 3):
//   R_dist = (exp(-0.5 * d / (c_d * w_dis)))^(1/w_den)
#pragma once

#include <RLGymSim_CPP/Utils/RewardFunctions/RewardFunction.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace RLbot {
using namespace RLGSC;

// ---------------------------------------------------------------------------
// Bausteine
// ---------------------------------------------------------------------------

// Kinesthetic Reward Combination: vorzeichenbehaftetes geometrisches Mittel.
// Anders als eine Summe kann eine einzelne schlechte Komponente nicht durch eine
// gute überdeckt werden: Ist eine Komponente 0, ist das Ergebnis 0.
class KRCReward : public RewardFunction {
public:
	std::vector<RewardFunction*> funcs;
	bool ownsFuncs;

	KRCReward(std::vector<RewardFunction*> funcs, bool ownsFuncs = true)
		: funcs(funcs), ownsFuncs(ownsFuncs) {
		RG_ASSERT(!funcs.empty());
	}
	RG_NO_COPY(KRCReward);

	virtual void Reset(const GameState& initialState) {
		for (auto f : funcs) f->Reset(initialState);
	}
	virtual void PreStep(const GameState& state) {
		for (auto f : funcs) f->PreStep(state);
	}
	virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
		double prod = 1.0;
		bool allPositive = true;
		for (auto f : funcs) {
			float r = f->GetReward(player, state, prevAction);
			if (r <= 0) allPositive = false;
			prod *= std::abs((double)r);
		}
		float mag = (float)std::pow(prod, 1.0 / (double)funcs.size());
		return allPositive ? mag : -mag;
	}
	virtual ~KRCReward() {
		if (ownsFuncs)
			for (auto f : funcs) delete f;
	}
};

// Parametrisierte Distanz-Belohnung, 1.0 bei Distanz 0 und monoton fallend.
// dispersion (w_dis) streckt die Kurve, density (w_den) macht sie steiler/flacher.
class ParamDistanceReward : public RewardFunction {
public:
	enum class Target { PLAYER_TO_BALL, BALL_TO_GOAL };

	Target target;
	float scale, dispersion, density;

	ParamDistanceReward(Target target, float scale, float dispersion = 1.f, float density = 1.f)
		: target(target), scale(scale), dispersion(dispersion), density(density) {}

	static float Compute(float dist, float scale, float dispersion, float density) {
		float base = std::exp(-0.5f * dist / (scale * dispersion));
		return std::pow(base, 1.f / density);
	}

	virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
		float dist;
		if (target == Target::PLAYER_TO_BALL) {
			// Abstand Oberfläche-zu-Mittelpunkt, damit Ballkontakt ~0 ergibt
			dist = (state.ball.pos - player.phys.pos).Length() - CommonValues::BALL_RADIUS;
		} else {
			Vec goal = (player.team == Team::BLUE) ? CommonValues::ORANGE_GOAL_BACK
			                                       : CommonValues::BLUE_GOAL_BACK;
			dist = (goal - state.ball.pos).Length() - CommonValues::BALL_RADIUS;
		}
		return Compute(RS_MAX(dist, 0.f), scale, dispersion, density);
	}
};

// Ausrichtung Spieler -> Ball -> gegnerisches Tor (Lucy: "Align Ball-to-Goal").
// defenseWeight > 0 belohnt zusätzlich die Position zwischen Ball und eigenem Tor.
class AlignBallGoalReward : public RewardFunction {
public:
	float offenseWeight, defenseWeight;

	AlignBallGoalReward(float offenseWeight = 1.f, float defenseWeight = 0.f)
		: offenseWeight(offenseWeight), defenseWeight(defenseWeight) {}

	virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
		Vec ownGoal = (player.team == Team::BLUE) ? CommonValues::BLUE_GOAL_BACK
		                                          : CommonValues::ORANGE_GOAL_BACK;
		Vec oppGoal = (player.team == Team::BLUE) ? CommonValues::ORANGE_GOAL_BACK
		                                          : CommonValues::BLUE_GOAL_BACK;
		Vec playerToBall = state.ball.pos - player.phys.pos;
		float len = playerToBall.Length();
		if (len < 1e-6f) return 0;
		Vec dir = playerToBall / len;

		float reward = 0;
		if (offenseWeight != 0)
			reward += offenseWeight * dir.Dot((oppGoal - state.ball.pos).Normalized());
		if (defenseWeight != 0)
			reward += defenseWeight * dir.Dot((state.ball.pos - ownGoal).Normalized());
		return reward;
	}
};

// Lucy ersetzt "Touch Ball Acceleration" durch "Touch Ball-to-Goal Acceleration":
// belohnt wird nur die Änderung der Ballgeschwindigkeit Richtung gegnerisches Tor,
// und nur bei einem eigenen Ballkontakt in diesem Step.
class TouchBallToGoalAccelReward : public RewardFunction {
public:
	// Geschwindigkeitsänderung Richtung Tor, normiert auf BALL_MAX_SPEED
	float lastVelToGoal[2] = { 0, 0 };
	float deltaVelToGoal[2] = { 0, 0 };
	bool hasLast = false;

	static float VelToGoal(const GameState& state, int team) {
		Vec goal = (team == (int)Team::BLUE) ? CommonValues::ORANGE_GOAL_BACK
		                                     : CommonValues::BLUE_GOAL_BACK;
		Vec dir = (goal - state.ball.pos).Normalized();
		return dir.Dot(state.ball.vel) / CommonValues::BALL_MAX_SPEED;
	}

	virtual void Reset(const GameState& initialState) {
		hasLast = false;
		for (int t = 0; t < 2; t++) {
			lastVelToGoal[t] = VelToGoal(initialState, t);
			deltaVelToGoal[t] = 0;
		}
	}

	virtual void PreStep(const GameState& state) {
		for (int t = 0; t < 2; t++) {
			float cur = VelToGoal(state, t);
			deltaVelToGoal[t] = hasLast ? (cur - lastVelToGoal[t]) : 0.f;
			lastVelToGoal[t] = cur;
		}
		hasLast = true;
	}

	virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
		if (!player.ballTouchedStep) return 0;
		return deltaVelToGoal[(int)player.team];
	}
};

// Reicht die Rewards eines Kindes unverändert durch und merkt sich die des letzten Schritts
// (Review-Befund R10). Sitzt zwischen den Lucy-Termen und einem ZeroSumReward: Im 1v1 mit
// Zero-Sum summieren sich die Rewards beider Spieler zu 0, "Average Step/Episode Reward" ist dann
// konstant 0 und sagt nichts mehr über das Shaping-Niveau. main.cpp loggt deshalb den Mittelwert
// von lastRaw als raw_step_reward (Metrics.h). Jedes Env hat seine eigene Instanz, Schreiben
// (Gym::Step) und Lesen (Step-Callback) passieren im selben Thread.
class RawRewardTap : public RewardFunction {
public:
	RewardFunction* child;
	std::vector<float> lastRaw;

	explicit RawRewardTap(RewardFunction* child) : child(child) {}
	RG_NO_COPY(RawRewardTap);

	virtual void Reset(const GameState& initialState) { child->Reset(initialState); }
	virtual void PreStep(const GameState& state) { child->PreStep(state); }
	virtual std::vector<float> GetAllRewards(const GameState& state, const ActionSet& prevActions, bool final) {
		lastRaw = child->GetAllRewards(state, prevActions, final);
		return lastRaw;
	}
	virtual ~RawRewardTap() { delete child; }
};

// Findet den RawRewardTap in einem von BuildLucyReward gebauten Reward (direkt oder als Kind
// des ZeroSumReward); nullptr, wenn es keinen gibt.
const RawRewardTap* FindRawRewardTap(const RewardFunction* fn);

// Belohnt Zeit in der Luft (nur wenn nicht am Boden und nicht demoliert).
class InAirReward : public RewardFunction {
public:
	virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction) {
		return (!player.carState.isOnGround && !player.carState.isDemoed) ? 1.f : 0.f;
	}
};

// Spieltest-Experimente (Default aus, nur per Config aktiv) ---------------------

// Erste Ballberührung nach einem Anstoß: 1 für den (tickgenau) ersten Berührer, sonst 0, einmal je
// Episode. Anstoß = Episode beginnt mit ruhendem Ball in der Mitte (wie PlayTracker). Mit Zero-Sum
// bekommt der andere Spieler im 1v1 dasselbe negativ.
class KickoffFirstTouchReward : public RewardFunction {
public:
	bool armed = false;
	uint64_t resetTick = 0;
	std::vector<uint32_t> winners;   // Car-IDs mit der ersten Berührung in diesem Schritt

	virtual void Reset(const GameState& initialState);
	virtual void PreStep(const GameState& state);
	virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction);
};

// Potenzialbasierte Form eines zustandsabhängigen Rewards Phi (Ng et al. 1999): r = gamma*Phi(s') - Phi(s).
// Über eine Episode summiert sich das (diskontiert) zu gamma^T*Phi(s_T) - Phi(s_0): Wer eine gute
// Lage nur hält, bekommt (gamma-1)*Phi, also leicht negativ; belohnt wird nur die Verbesserung.
// Am Tor (echtes Episodenende) ist Phi(s') = 0; bei Truncation (Zeitlimits) läuft die Welt weiter,
// dort gilt Phi(s'). Phi muss zustandslos sein (die KRC-Terme sind es).
class PotentialReward : public RewardFunction {
public:
	RewardFunction* phi;
	float gamma;
	std::unordered_map<uint32_t, float> last;

	PotentialReward(RewardFunction* phi, float gamma) : phi(phi), gamma(gamma) {}
	RG_NO_COPY(PotentialReward);
	virtual ~PotentialReward() { delete phi; }

	virtual void Reset(const GameState& initialState);
	virtual void PreStep(const GameState& state) { phi->PreStep(state); }
	virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction);
	virtual float GetFinalReward(const PlayerData& player, const GameState& state, const Action& prevAction);
private:
	float Step(const PlayerData& player, float next);
};

// Ballberührung in der Luft, skaliert mit der Ballhöhe beim Kontakt: (z - Ballradius) / (Decke -
// Ballradius), also ~0,05 für einen Sprung an den rollenden Ball und ~0,5 bei 1000 uu. Nur wenn das
// Auto nicht auf dem Boden ist. Je Auto höchstens einmal in COOLDOWN_TICKS (0,5 s), damit
// Dauerkontakt (Luftdribbling) nicht pro Schritt zahlt.
class AirTouchReward : public RewardFunction {
public:
	static constexpr uint64_t COOLDOWN_TICKS = 60;
	std::unordered_map<uint32_t, uint64_t> lastRewardTick;

	virtual void Reset(const GameState& initialState) { lastRewardTick.clear(); }
	virtual float GetReward(const PlayerData& player, const GameState& state, const Action& prevAction);
};

// ---------------------------------------------------------------------------
// Zusammenbau
// ---------------------------------------------------------------------------

// Gewichte einer Reward-Phase (Bauplan §6, Phase A/B/C).
struct RewardWeights {
	float goal = 10.f;
	float concede = 10.f;          // wird als negatives Event gewertet
	float demo = 0.f;
	float demoed = 0.f;
	float touchBallToGoalAccel = 1.f;
	float offensivePotential = 1.f;
	float distWeightedAlign = 0.5f;
	float velocityPlayerToBall = 0.1f;
	float saveBoost = 0.3f;
	float inAir = 0.02f;
	float teamSpirit = 0.f;        // 0 = rein eigennützig, 1 = volle Teamteilung
	// Spieltest-Experimente, 0 = aus (bisheriges Verhalten)
	float kickoffFirstTouch = 0.f;         // KickoffFirstTouchReward
	float airTouch = 0.f;                  // AirTouchReward
	// > 0: offensive_potential_krc und dist_weighted_align_krc als PotentialReward mit diesem
	// Zusatzfaktor (Gewicht * Faktor); 0 = direkte Form wie bisher
	float potentialShapingScale = 0.f;
	float potentialGamma = 0.9954f;        // = learner.gae_gamma (setzt TrainConfig)
};

// KRC(Align, VelocityPlayerToBall, Distance) - "Offensive Potential"
RewardFunction* MakeOffensivePotential();
// KRC(Align, Distance) - "Distance-weighted Alignment"
RewardFunction* MakeDistWeightedAlignment();

// Ein Summand des Lucy-Rewards mit Namen (für die Reward-Bilanz, tools/cpp/reward_budget.cpp).
struct RewardPart {
	std::string name;
	RewardFunction* fn;
	float weight;
};

// Die Summanden in der Reihenfolge, in der BuildLucyReward sie kombiniert (nur Gewichte != 0).
// Der Aufrufer übernimmt die Objekte.
std::vector<RewardPart> BuildLucyRewardParts(const RewardWeights& w);

// Baut den vollständigen Lucy-nahen Reward. Der Aufrufer übernimmt das Objekt.
RewardFunction* BuildLucyReward(const RewardWeights& w);

} // namespace RLbot
