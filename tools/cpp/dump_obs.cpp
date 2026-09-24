// Erzeugt Golden-Fixtures für den Obs-Paritätstest zwischen C++ (Training) und Python (Deploy).
//
//   dump_obs.exe <out.json> [anzahl_faelle]
//
// Jeder Fall enthält den vollständigen Spielzustand, die Aktionshistorie und den daraus
// gebauten Obs-Vektor pro Spieler. tests/test_obs_parity.py baut dieselben Vektoren mit
// env/obs_python.py nach und vergleicht sie.
#include "env/cpp/Obs.h"

#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <random>

using namespace RLGSC;
using namespace RLbot;
using nlohmann::json;

static std::mt19937 rng(20260923);

static float Rand(float min, float max) {
	return std::uniform_real_distribution<float>(min, max)(rng);
}
static bool RandBool(float p = 0.5f) { return Rand(0, 1) < p; }

static json VecToJSON(Vec v) { return json::array({ v.x, v.y, v.z }); }

int main(int argc, char** argv) {
	if (argc < 2) {
		std::cerr << "usage: dump_obs <out.json> [count]\n";
		return 2;
	}
	int count = argc > 2 ? std::stoi(argv[2]) : 30;

	constexpr int MAX_PLAYERS = 3;
	constexpr int STACK = 5;

	json out;
	out["obs_size"] = StackedPaddedOBS::GetOBSSize(MAX_PLAYERS, STACK);
	out["max_players"] = MAX_PLAYERS;
	out["action_stack_size"] = STACK;
	out["player_features"] = StackedPaddedOBS::PLAYER_FEATURES;
	out["ball_features"] = StackedPaddedOBS::BALL_FEATURES;
	out["cases"] = json::array();

	// Aktionstabelle mitschreiben: Die Policy gibt einen Index aus, Training und Deployment
	// müssen darunter exakt dieselbe Eingabe verstehen.
	{
		DiscreteAction parser;
		out["action_table"] = json::array();
		for (auto& action : parser.actions) {
			json ja = json::array();
			for (int e = 0; e < (int)Action::ELEM_AMOUNT; e++)
				ja.push_back(action[e]);
			out["action_table"].push_back(ja);
		}
	}

	// Rotations-Testfälle: Euler-Winkel und die daraus entstehenden Richtungsvektoren.
	// Beim Deployment liefert RLBot Euler-Winkel, das Obs braucht forward/up.
	{
		out["rotation_cases"] = json::array();
		std::vector<std::array<float, 3>> angles = {
			{ 0, 0, 0 }, { 0.5f, 0, 0 }, { 0, 0.5f, 0 }, { 0, 0, 0.5f },
			{ 0.3f, -1.2f, 2.5f }, { -1.5f, 3.0f, -3.0f }, { 1.2f, 0.7f, -0.4f },
		};
		for (int i = 0; i < 40; i++)
			angles.push_back({ Rand(-M_PI / 2, M_PI / 2), Rand(-M_PI, M_PI), Rand(-M_PI, M_PI) });

		for (auto& a : angles) {
			RotMat mat = Angle(a[1], a[0], a[2]).ToRotMat(); // Angle(yaw, pitch, roll)
			out["rotation_cases"].push_back({
				{ "pitch", a[0] }, { "yaw", a[1] }, { "roll", a[2] },
				{ "forward", VecToJSON(mat.forward) },
				{ "right", VecToJSON(mat.right) },
				{ "up", VecToJSON(mat.up) },
			});
		}
	}

	for (int caseIdx = 0; caseIdx < count; caseIdx++) {
		int teamSize = (caseIdx % 3) + 1;

		GameState state = {};
		state.ball.pos = Vec(Rand(-3000, 3000), Rand(-4500, 4500), Rand(93, 1800));
		state.ball.vel = Vec(Rand(-3000, 3000), Rand(-3000, 3000), Rand(-1500, 1500));
		state.ball.angVel = Vec(Rand(-6, 6), Rand(-6, 6), Rand(-6, 6));
		state.ball.rotMat = Angle(Rand(-M_PI, M_PI), Rand(-M_PI / 2, M_PI / 2), Rand(-M_PI, M_PI)).ToRotMat();
		state.ballInv = state.ball.Invert();

		for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
			float timer = RandBool(0.4f) ? Rand(0, 10) : 0.f;
			state.boostPadTimers[i] = timer;
			state.boostPads[i] = timer == 0;
		}
		for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
			// RocketSim spiegelt die Pad-Reihenfolge; die Liste ist paarweise symmetrisch.
			state.boostPadTimersInv[i] = state.boostPadTimers[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1];
			state.boostPadsInv[i] = state.boostPads[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1];
		}

		for (int t = 0; t < 2; t++) {
			for (int i = 0; i < teamSize; i++) {
				PlayerData p = {};
				p.carId = t * 100 + i + 1;
				p.team = t == 0 ? Team::BLUE : Team::ORANGE;
				p.phys.pos = Vec(Rand(-3800, 3800), Rand(-5000, 5000), Rand(17, 1900));
				p.phys.vel = Vec(Rand(-2300, 2300), Rand(-2300, 2300), Rand(-2300, 2300));
				p.phys.angVel = Vec(Rand(-5.5f, 5.5f), Rand(-5.5f, 5.5f), Rand(-5.5f, 5.5f));
				p.phys.rotMat = Angle(Rand(-M_PI, M_PI), Rand(-M_PI / 2, M_PI / 2), Rand(-M_PI, M_PI)).ToRotMat();
				p.physInv = p.phys.Invert();
				p.boostFraction = Rand(0, 1);
				p.hasFlip = RandBool();
				p.hasJump = RandBool();
				p.carState.isOnGround = RandBool();
				p.carState.isDemoed = RandBool(0.1f);
				p.carState.isSupersonic = RandBool(0.3f);
				p.carState.isFlipping = RandBool(0.2f);
				p.carState.isJumping = RandBool(0.2f);
				state.players.push_back(p);
			}
		}

		// Aktionshistorie: STACK Aktionen pro Spieler, älteste zuerst
		std::vector<std::vector<Action>> history(state.players.size());
		for (size_t pi = 0; pi < state.players.size(); pi++)
			for (int k = 0; k < STACK; k++) {
				Action a = {};
				for (int e = 0; e < (int)Action::ELEM_AMOUNT; e++)
					a[e] = std::round(Rand(-1, 1) * 100.f) / 100.f;
				history[pi].push_back(a);
			}

		// Shuffle aus, damit die Reihenfolge deterministisch vergleichbar ist
		StackedPaddedOBS obs(MAX_PLAYERS, STACK, false);
		obs.Reset(state);
		std::vector<FList> finalObs(state.players.size());
		for (int k = 0; k < STACK; k++)
			for (size_t pi = 0; pi < state.players.size(); pi++)
				finalObs[pi] = obs.BuildOBS(state.players[pi], state, history[pi][k]);

		json c;
		c["team_size"] = teamSize;
		c["ball"] = {
			{ "pos", VecToJSON(state.ball.pos) },
			{ "vel", VecToJSON(state.ball.vel) },
			{ "ang_vel", VecToJSON(state.ball.angVel) },
		};
		c["pad_timers"] = json::array();
		c["pad_timers_inv"] = json::array();
		for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
			c["pad_timers"].push_back(state.boostPadTimers[i]);
			c["pad_timers_inv"].push_back(state.boostPadTimersInv[i]);
		}

		c["players"] = json::array();
		for (size_t pi = 0; pi < state.players.size(); pi++) {
			auto& p = state.players[pi];
			json jp = {
				{ "car_id", p.carId },
				{ "team", (int)p.team },
				{ "pos", VecToJSON(p.phys.pos) },
				{ "forward", VecToJSON(p.phys.rotMat.forward) },
				{ "up", VecToJSON(p.phys.rotMat.up) },
				{ "vel", VecToJSON(p.phys.vel) },
				{ "ang_vel", VecToJSON(p.phys.angVel) },
				{ "boost", p.boostFraction },
				{ "on_ground", p.carState.isOnGround },
				{ "has_flip", p.hasFlip },
				{ "has_jump", p.hasJump },
				{ "demoed", p.carState.isDemoed },
				{ "supersonic", p.carState.isSupersonic },
				{ "flipping", p.carState.isFlipping },
				{ "jumping", p.carState.isJumping },
			};
			jp["action_history"] = json::array();
			for (auto& a : history[pi]) {
				json ja = json::array();
				for (int e = 0; e < (int)Action::ELEM_AMOUNT; e++)
					ja.push_back(a[e]);
				jp["action_history"].push_back(ja);
			}
			jp["obs"] = finalObs[pi];
			c["players"].push_back(jp);
		}
		out["cases"].push_back(c);
	}

	std::ofstream fOut(argv[1]);
	if (!fOut.good()) {
		std::cerr << "Kann " << argv[1] << " nicht schreiben\n";
		return 1;
	}
	fOut << out.dump(1);
	std::cout << "Geschrieben: " << argv[1] << " (" << count << " Fälle, Obs-Größe "
	          << StackedPaddedOBS::GetOBSSize(MAX_PLAYERS, STACK) << ")\n";
	return 0;
}
