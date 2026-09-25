#include "Obs.h"

#include <RLGymSim_CPP/Math.h>

namespace RLbot {

void StackedPaddedOBS::AddPlayerToOBS(FList& obs, const PlayerData& player, const PhysObj& ball, bool inv) const {
	const PhysObj& phys = player.GetPhys(inv);
	const CarState& cs = player.carState;

	obs += phys.pos * posCoef;
	obs += phys.rotMat.forward;
	obs += phys.rotMat.up;
	obs += phys.vel * velCoef;
	obs += phys.angVel * angVelCoef;
	obs += {
		player.boostFraction,
		(float)cs.isOnGround,
		(float)player.hasFlip,
		(float)player.hasJump,
		(float)cs.isDemoed,
		(float)cs.isSupersonic,
		(float)cs.isFlipping,
		(float)cs.isJumping,
	};
	obs += (ball.pos - phys.pos) * posCoef;
	obs += (ball.vel - phys.vel) * velCoef;
}

FList StackedPaddedOBS::BuildOBS(const PlayerData& player, const GameState& state, const Action& prevAction) {
	FList result = {};
	result.reserve(GetOBSSize());

	bool inv = player.team == Team::ORANGE;
	const PhysObj& ball = state.GetBallPhys(inv);
	const auto& padTimers = inv ? state.boostPadTimersInv : state.boostPadTimers;

	result += ball.pos * posCoef;
	result += ball.vel * velCoef;
	result += ball.angVel * angVelCoef;
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++)
		result += padTimers[i] * padTimerCoef;

	AddPlayerToOBS(result, player, ball, inv);

	{ // Aktions-Stack: älteste zuerst, mit Nullen aufgefüllt
		auto& history = actionHistory[player.carId];
		history.push_back(prevAction);
		while ((int)history.size() > actionStackSize)
			history.pop_front();

		int padding = actionStackSize - (int)history.size();
		for (int i = 0; i < padding * (int)Action::ELEM_AMOUNT; i++)
			result += 0.f;
		for (const Action& act : history)
			for (int i = 0; i < (int)Action::ELEM_AMOUNT; i++)
				result += act[i];
	}

	FList2 teammates = {}, opponents = {};
	for (auto& other : state.players) {
		if (other.carId == player.carId)
			continue;
		FList playerObs = {};
		AddPlayerToOBS(playerObs, other, ball, inv);
		((other.team == player.team) ? teammates : opponents).push_back(playerObs);
	}

	if ((int)teammates.size() > maxPlayers - 1)
		RG_ERR_CLOSE("StackedPaddedOBS: zu viele Mitspieler, Maximum ist " << (maxPlayers - 1));
	if ((int)opponents.size() > maxPlayers)
		RG_ERR_CLOSE("StackedPaddedOBS: zu viele Gegner, Maximum ist " << maxPlayers);

	for (int i = 0; i < 2; i++) {
		FList2& list = i ? opponents : teammates;
		int target = i ? maxPlayers : maxPlayers - 1;
		while ((int)list.size() < target)
			list.push_back(FList(PLAYER_FEATURES, 0.f));
		if (shuffle)
			ShuffleSlots(list);
	}

	for (auto& teammate : teammates)
		result += teammate;
	for (auto& opponent : opponents)
		result += opponent;

	RG_ASSERT((int)result.size() == GetOBSSize());
	return result;
}

void StackedPaddedOBS::ShuffleSlots(FList2& list) {
	if (shuffleSeed >= 0)
		std::shuffle(list.begin(), list.end(), rng);
	else
		std::shuffle(list.begin(), list.end(), ::Math::GetRandEngine());
}

} // namespace RLbot
