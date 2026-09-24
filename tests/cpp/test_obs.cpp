// Prüft Layout, Padding, Aktions-Stack und Team-Inversion des Obs-Builders.
#include "test_util.h"

#include "env/cpp/Obs.h"

#include <algorithm>
#include <numeric>

using namespace RLGSC;
using namespace RLbot;

static PlayerData MakePlayer(uint32_t id, Team team, Vec pos, Vec vel = {}, float boost = 0.5f) {
	PlayerData p = {};
	p.carId = id;
	p.team = team;
	p.phys.pos = pos;
	p.phys.vel = vel;
	p.phys.rotMat = RotMat::GetIdentity();
	p.physInv = p.phys.Invert();
	p.carState.isOnGround = true;
	p.boostFraction = boost;
	p.hasFlip = true;
	p.hasJump = true;
	return p;
}

static GameState MakeState(std::vector<PlayerData> players, Vec ballPos, Vec ballVel = {}) {
	GameState s = {};
	s.players = players;
	s.ball.pos = ballPos;
	s.ball.vel = ballVel;
	s.ball.rotMat = RotMat::GetIdentity();
	s.ballInv = s.ball.Invert();
	s.boostPads.fill(true);
	s.boostPadsInv.fill(true);
	s.boostPadTimers.fill(0);
	s.boostPadTimersInv.fill(0);
	return s;
}

static GameState MakeMatch(int teamSize, Vec ballPos = Vec(0, 0, 93)) {
	std::vector<PlayerData> players;
	for (int i = 0; i < teamSize; i++)
		players.push_back(MakePlayer(i + 1, Team::BLUE, Vec(-1000.f - 100 * i, -2000, 17)));
	for (int i = 0; i < teamSize; i++)
		players.push_back(MakePlayer(100 + i, Team::ORANGE, Vec(1000.f + 100 * i, 2000, 17)));
	return MakeState(players, ballPos);
}

// --- Größe und Team-Unabhängigkeit --------------------------------------

TEST(OBS_Groesse_ist_teamgroessenunabhaengig) {
	CHECK_EQ(StackedPaddedOBS::GetOBSSize(3, 5), 257);

	for (int teamSize = 1; teamSize <= 3; teamSize++) {
		StackedPaddedOBS obs(3, 5, false);
		auto state = MakeMatch(teamSize);
		obs.Reset(state);
		Action empty = {};
		auto vec = obs.BuildOBS(state.players[0], state, empty);
		CHECK_EQ((int)vec.size(), 257);
	}
}

TEST(OBS_Groessenformel_stimmt_mit_Ausgabe) {
	for (int maxPlayers = 1; maxPlayers <= 3; maxPlayers++) {
		for (int k = 1; k <= 5; k++) {
			StackedPaddedOBS obs(maxPlayers, k, false);
			auto state = MakeMatch(1);
			obs.Reset(state);
			Action empty = {};
			auto vec = obs.BuildOBS(state.players[0], state, empty);
			CHECK_EQ((int)vec.size(), StackedPaddedOBS::GetOBSSize(maxPlayers, k));
		}
	}
}

// --- Layout --------------------------------------------------------------

TEST(OBS_Ball_und_Padtimer_Block) {
	StackedPaddedOBS obs(3, 5, false);
	auto state = MakeMatch(1, Vec(2048, 2560, 1022));
	state.ball.vel = Vec(1150, 0, 0);
	state.ballInv = state.ball.Invert();
	state.boostPadTimers.fill(5.f);
	state.boostPadTimersInv.fill(5.f);

	obs.Reset(state);
	Action empty = {};
	auto vec = obs.BuildOBS(state.players[0], state, empty);

	CHECK_NEAR(vec[0], 2048.f / CommonValues::SIDE_WALL_X, 1e-6);   // 0.5
	CHECK_NEAR(vec[1], 2560.f / CommonValues::BACK_WALL_Y, 1e-6);   // 0.5
	CHECK_NEAR(vec[2], 1022.f / CommonValues::CEILING_Z, 1e-6);     // 0.5
	CHECK_NEAR(vec[3], 1150.f / CommonValues::CAR_MAX_SPEED, 1e-6); // 0.5
	CHECK_NEAR(vec[4], 0.0, 1e-9);
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++)
		CHECK_NEAR(vec[9 + i], 0.5, 1e-6);                          // 5 s * (1/10)
}

TEST(OBS_Selbst_Block_direkt_nach_den_Padtimern) {
	StackedPaddedOBS obs(3, 5, false);
	auto state = MakeState({ MakePlayer(1, Team::BLUE, Vec(-2048, -2560, 17), Vec(1150, 0, 0), 0.75f) },
	                       Vec(0, 0, 93));
	obs.Reset(state);
	Action empty = {};
	auto vec = obs.BuildOBS(state.players[0], state, empty);

	int self = StackedPaddedOBS::BALL_FEATURES + CommonValues::BOOST_LOCATIONS_AMOUNT; // 43
	CHECK_NEAR(vec[self + 0], -0.5, 1e-6);
	CHECK_NEAR(vec[self + 1], -0.5, 1e-6);
	CHECK_NEAR(vec[self + 2], 17.f / CommonValues::CEILING_Z, 1e-6);
	CHECK_NEAR(vec[self + 3], 1.0, 1e-6);    // forward.x (Identität)
	CHECK_NEAR(vec[self + 8], 1.0, 1e-6);    // up.z
	CHECK_NEAR(vec[self + 9], 0.5, 1e-6);    // vel.x / CAR_MAX_SPEED
	CHECK_NEAR(vec[self + 15], 0.75, 1e-6);  // boostFraction
	CHECK_NEAR(vec[self + 16], 1.0, 1e-9);   // isOnGround
	CHECK_NEAR(vec[self + 19], 0.0, 1e-9);   // isDemoed
	// Relativposition zum Ball
	CHECK_NEAR(vec[self + 23], 2048.f / CommonValues::SIDE_WALL_X, 1e-6);
	CHECK_NEAR(vec[self + 24], 2560.f / CommonValues::BACK_WALL_Y, 1e-6);
}

// --- Aktions-Stack -------------------------------------------------------

TEST(OBS_Aktionsstack_schiebt_durch) {
	StackedPaddedOBS obs(3, 5, false);
	auto state = MakeMatch(1);
	obs.Reset(state);

	int stackStart = StackedPaddedOBS::BALL_FEATURES + CommonValues::BOOST_LOCATIONS_AMOUNT
		+ StackedPaddedOBS::PLAYER_FEATURES;
	int elems = (int)Action::ELEM_AMOUNT;

	// Erster Aufruf: 4 Nullblöcke, dann die übergebene Aktion
	Action a1 = {}; a1.throttle = 1.f;
	auto v1 = obs.BuildOBS(state.players[0], state, a1);
	for (int i = 0; i < 4 * elems; i++)
		CHECK_NEAR(v1[stackStart + i], 0.0, 1e-9);
	CHECK_NEAR(v1[stackStart + 4 * elems], 1.0, 1e-9);

	// Zweiter Aufruf: a1 rutscht eine Position nach vorn
	Action a2 = {}; a2.throttle = 2.f;
	auto v2 = obs.BuildOBS(state.players[0], state, a2);
	CHECK_NEAR(v2[stackStart + 3 * elems], 1.0, 1e-9);
	CHECK_NEAR(v2[stackStart + 4 * elems], 2.0, 1e-9);

	// Nach sechs Aufrufen ist a1 herausgefallen
	for (int i = 3; i <= 6; i++) {
		Action a = {}; a.throttle = (float)i;
		obs.BuildOBS(state.players[0], state, a);
	}
	Action a7 = {}; a7.throttle = 7.f;
	auto v7 = obs.BuildOBS(state.players[0], state, a7);
	for (int slot = 0; slot < 5; slot++)
		CHECK_NEAR(v7[stackStart + slot * elems], 3.0 + slot, 1e-9);

	// Reset leert die Historie wieder
	obs.Reset(state);
	auto v8 = obs.BuildOBS(state.players[0], state, a1);
	for (int i = 0; i < 4 * elems; i++)
		CHECK_NEAR(v8[stackStart + i], 0.0, 1e-9);
}

TEST(OBS_Aktionsstack_ist_pro_Spieler_getrennt) {
	StackedPaddedOBS obs(3, 5, false);
	auto state = MakeMatch(1);
	obs.Reset(state);

	int stackStart = StackedPaddedOBS::BALL_FEATURES + CommonValues::BOOST_LOCATIONS_AMOUNT
		+ StackedPaddedOBS::PLAYER_FEATURES;
	int elems = (int)Action::ELEM_AMOUNT;

	Action blue = {}; blue.throttle = 1.f;
	Action orange = {}; orange.throttle = -1.f;
	obs.BuildOBS(state.players[0], state, blue);
	auto vOrange = obs.BuildOBS(state.players[1], state, orange);

	// Der erste Aufruf für Orange darf nur dessen eigene Aktion enthalten
	for (int i = 0; i < 4 * elems; i++)
		CHECK_NEAR(vOrange[stackStart + i], 0.0, 1e-9);
	CHECK_NEAR(vOrange[stackStart + 4 * elems], -1.0, 1e-9);
}

// --- Padding und Slots ---------------------------------------------------

TEST(OBS_Padding_fuellt_leere_Slots_mit_Null) {
	StackedPaddedOBS obs(3, 5, false);
	auto state = MakeMatch(1);   // 1v1: 0 Mitspieler, 1 Gegner
	obs.Reset(state);
	Action empty = {};
	auto vec = obs.BuildOBS(state.players[0], state, empty);

	int P = StackedPaddedOBS::PLAYER_FEATURES;
	int mates = StackedPaddedOBS::BALL_FEATURES + CommonValues::BOOST_LOCATIONS_AMOUNT + P + 5 * 8;

	// Beide Mitspieler-Slots leer
	for (int i = 0; i < 2 * P; i++)
		CHECK_NEAR(vec[mates + i], 0.0, 1e-9);

	// Erster Gegner-Slot gefüllt, die beiden anderen leer
	int opps = mates + 2 * P;
	bool anyNonZero = false;
	for (int i = 0; i < P; i++)
		anyNonZero |= std::abs(vec[opps + i]) > 1e-9;
	CHECK(anyNonZero);
	for (int i = P; i < 3 * P; i++)
		CHECK_NEAR(vec[opps + i], 0.0, 1e-9);
}

TEST(OBS_Shuffle_erhaelt_die_Menge_der_Gegner) {
	StackedPaddedOBS obs(3, 5, true);
	auto state = MakeMatch(3);
	obs.Reset(state);
	Action empty = {};

	int P = StackedPaddedOBS::PLAYER_FEATURES;
	int opps = StackedPaddedOBS::BALL_FEATURES + CommonValues::BOOST_LOCATIONS_AMOUNT + P + 5 * 8 + 2 * P;

	// Referenz ohne Shuffle
	StackedPaddedOBS fixedObs(3, 5, false);
	fixedObs.Reset(state);
	auto ref = fixedObs.BuildOBS(state.players[0], state, empty);
	std::vector<float> refX;
	for (int slot = 0; slot < 3; slot++)
		refX.push_back(ref[opps + slot * P]);
	std::sort(refX.begin(), refX.end());

	for (int trial = 0; trial < 20; trial++) {
		auto vec = obs.BuildOBS(state.players[0], state, empty);
		std::vector<float> gotX;
		for (int slot = 0; slot < 3; slot++)
			gotX.push_back(vec[opps + slot * P]);
		std::sort(gotX.begin(), gotX.end());
		for (size_t i = 0; i < refX.size(); i++)
			CHECK_NEAR(gotX[i], refX[i], 1e-6);
	}
}

// --- Team-Inversion ------------------------------------------------------

TEST(OBS_Orange_sieht_gespiegelte_Welt) {
	// Spiegelbildliche Aufstellung: Blau bei -Y, Orange bei +Y, Ball in der Mitte.
	// Beide müssen exakt dieselbe Beobachtung bekommen.
	auto state = MakeState({
		MakePlayer(1, Team::BLUE, Vec(-800, -2000, 17), Vec(0, 500, 0), 0.4f),
		MakePlayer(2, Team::ORANGE, Vec(800, 2000, 17), Vec(0, -500, 0), 0.4f),
	}, Vec(0, 0, 93));
	// Orange-Auto um 180 Grad gedreht, damit auch die Rotation spiegelbildlich ist
	state.players[1].phys.rotMat = Angle(M_PI, 0, 0).ToRotMat();
	state.players[1].physInv = state.players[1].phys.Invert();

	StackedPaddedOBS obs(3, 5, false);
	obs.Reset(state);
	Action empty = {};
	auto blue = obs.BuildOBS(state.players[0], state, empty);
	auto orange = obs.BuildOBS(state.players[1], state, empty);

	CHECK_EQ(blue.size(), orange.size());
	for (size_t i = 0; i < blue.size(); i++)
		CHECK_NEAR(orange[i], blue[i], 1e-5);
}
