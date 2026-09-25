// Team-size-agnostischer Obs-Builder mit Padding, Slot-Shuffling, Boost-Pad-Timern
// und k gestapelten vorherigen Aktionen (Bauplan v2 §5, Lucy k=5).
//
// Layout (alles aus Sicht des Spielers, für Orange invertiert):
//   [0]                  ball.pos   * posCoef            (3)
//                        ball.vel   * velCoef            (3)
//                        ball.angVel* angVelCoef         (3)
//                        boostPadTimer * padTimerCoef    (34)
//   dann                 Spieler-Block "self"            (PLAYER_FEATURES)
//   dann                 k vorherige Aktionen, alt->neu  (k * 8)
//   dann                 (maxPlayers-1) Mitspieler-Blöcke, gepolstert + gemischt
//   dann                 maxPlayers Gegner-Blöcke,        gepolstert + gemischt
//
// Spieler-Block (PLAYER_FEATURES = 29):
//   pos*posCoef (3), forward (3), up (3), vel*velCoef (3), angVel*angVelCoef (3),
//   boostFraction, isOnGround, hasFlip, hasJump, isDemoed, isSupersonic, isFlipping, isJumping (8),
//   (ball.pos - pos)*posCoef (3), (ball.vel - vel)*velCoef (3)
//
// Dieses Layout muss zwischen Training (C++) und Deployment (Python) bitgleich sein,
// siehe env/obs_python.py und tests/test_obs_parity.py.
#pragma once

#include <RLGymSim_CPP/Utils/OBSBuilders/OBSBuilder.h>

#include <deque>
#include <random>
#include <unordered_map>

namespace RLbot {
using namespace RLGSC;

class StackedPaddedOBS : public OBSBuilder {
public:
	constexpr static int PLAYER_FEATURES = 29;
	constexpr static int BALL_FEATURES = 9;

	int maxPlayers;
	int actionStackSize;
	// Slot-Shuffle pro BuildOBS (Audit H3). Im Training über env.shuffle_slots schaltbar,
	// im Deployment (env/obs_python.py) und im Duell immer aus.
	bool shuffle;

	Vec posCoef;
	float velCoef, angVelCoef, padTimerCoef;

	std::unordered_map<uint32_t, std::deque<Action>> actionHistory;

	// Seed des eigenen Shuffle-RNGs (Audit H6); -1 = RocketSims globaler, zeitgeseedeter Engine.
	int64_t shuffleSeed;

	StackedPaddedOBS(
		int maxPlayers = 3,
		int actionStackSize = 5,
		bool shuffle = true,
		Vec posCoef = Vec(1 / CommonValues::SIDE_WALL_X, 1 / CommonValues::BACK_WALL_Y, 1 / CommonValues::CEILING_Z),
		float velCoef = 1 / CommonValues::CAR_MAX_SPEED,
		float angVelCoef = 1 / CommonValues::CAR_MAX_ANG_VEL,
		float padTimerCoef = 1 / 10.f,
		int64_t shuffleSeed = -1
	) : maxPlayers(maxPlayers), actionStackSize(actionStackSize), shuffle(shuffle),
		posCoef(posCoef), velCoef(velCoef), angVelCoef(angVelCoef), padTimerCoef(padTimerCoef),
		shuffleSeed(shuffleSeed), rng(shuffleSeed >= 0 ? (uint64_t)shuffleSeed : 0) {}

	static int GetOBSSize(int maxPlayers, int actionStackSize) {
		return BALL_FEATURES + CommonValues::BOOST_LOCATIONS_AMOUNT
			+ PLAYER_FEATURES                                  // self
			+ actionStackSize * (int)Action::ELEM_AMOUNT       // Aktions-Stack
			+ (maxPlayers - 1) * PLAYER_FEATURES               // Mitspieler
			+ maxPlayers * PLAYER_FEATURES;                    // Gegner
	}
	int GetOBSSize() const { return GetOBSSize(maxPlayers, actionStackSize); }

	void AddPlayerToOBS(FList& obs, const PlayerData& player, const PhysObj& ball, bool inv) const;

	virtual void Reset(const GameState& initialState) { actionHistory.clear(); }
	virtual FList BuildOBS(const PlayerData& player, const GameState& state, const Action& prevAction);

private:
	std::mt19937_64 rng;
	void ShuffleSlots(FList2& list);
};

} // namespace RLbot
