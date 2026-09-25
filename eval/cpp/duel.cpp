// Lässt zwei Checkpoints in RocketSim gegeneinander spielen und schreibt das Ergebnis als JSON.
//
//   duel.exe --a <PPO_POLICY.lt> --b <PPO_POLICY.lt> --games 100 --team-size 1 --out result.json
//
// Die Seiten werden jedes Spiel getauscht, damit Kickoff- und Seitenvorteile sich aufheben.
// Gezählt werden Tore (varianzärmer als Sieg/Niederlage, siehe Bauplan §8).
#include "policy_io.h"

#include "env/cpp/Obs.h"
#include "env/cpp/StateSetters.h"
#include "env/cpp/TimeoutCondition.h"

#include <RLGymSim_CPP/Gym.h>
#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/StateSetters/RandomState.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <fstream>
#include <iostream>

using namespace RLGSC;
using namespace RLbot;
using nlohmann::json;

struct Args {
	std::string pathA, pathB, outPath = "duel_result.json";
	std::string meshDir = "collision_meshes";
	int games = 50, teamSize = 1, tickSkip = 8, maxSeconds = 120, actionStack = 5, maxPlayers = 3;
	bool deterministic = false;
	float temperature = 1.f;
	int seed = 123;
	// kickoff ist der Standard für vergleichbare Ratings; defense/random dienen dazu,
	// die Torzählung zu prüfen und gezielte Szenen zu bewerten.
	std::string setter = "kickoff";
};

static StateSetter* MakeSetter(const std::string& name) {
	if (name == "kickoff") return new KickoffSetter();
	if (name == "defense") return new DefenseSetter();
	if (name == "random") return new RandomState(true, true, true);
	if (name == "aerial") return new AerialSetter();
	std::cerr << "Unbekannter State-Setter: " << name << "\n";
	exit(2);
}

int main(int argc, char** argv) {
	Args args;
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		auto next = [&]() -> std::string {
			if (i + 1 >= argc) { std::cerr << "Fehlender Wert für " << arg << "\n"; exit(2); }
			return argv[++i];
		};
		if (arg == "--a") args.pathA = next();
		else if (arg == "--b") args.pathB = next();
		else if (arg == "--out") args.outPath = next();
		else if (arg == "--meshes") args.meshDir = next();
		else if (arg == "--games") args.games = std::stoi(next());
		else if (arg == "--team-size") args.teamSize = std::stoi(next());
		else if (arg == "--max-seconds") args.maxSeconds = std::stoi(next());
		else if (arg == "--action-stack") args.actionStack = std::stoi(next());
		else if (arg == "--max-players") args.maxPlayers = std::stoi(next());
		else if (arg == "--deterministic") args.deterministic = true;
		else if (arg == "--temperature") args.temperature = std::stof(next());
		else if (arg == "--seed") args.seed = std::stoi(next());
		else if (arg == "--setter") args.setter = next();
		else { std::cerr << "Unbekanntes Argument: " << arg << "\n"; return 2; }
	}
	if (args.pathA.empty() || args.pathB.empty()) {
		std::cerr << "usage: duel --a <policy.lt> --b <policy.lt> [--games N] [--team-size N] "
		             "[--deterministic] [--out result.json]\n";
		return 2;
	}

	torch::NoGradGuard noGrad;
	torch::manual_seed(args.seed);

	LoadedPolicy a, b;
	try {
		a = LoadPolicy(args.pathA);
		b = LoadPolicy(args.pathB);
	} catch (const std::exception& e) {
		std::cerr << e.what() << "\n";
		return 1;
	}
	int expectedObs = StackedPaddedOBS::GetOBSSize(args.maxPlayers, args.actionStack);
	for (auto* p : { &a, &b }) {
		if (p->obsSize != expectedObs) {
			std::cerr << "Policy erwartet Obs-Größe " << p->obsSize << ", der Obs-Builder liefert "
			          << expectedObs << ". Passen max_players/action_stack_size zum Training?\n";
			return 1;
		}
	}

	RocketSim::Init(args.meshDir);

	auto* obsBuilder = new StackedPaddedOBS(args.maxPlayers, args.actionStack, false);
	auto* parser = new DiscreteAction();
	auto* match = new Match(
		new EventReward({}),
		{ new GoalScoreCondition(), new TimeoutCondition(args.maxSeconds * 120 / args.tickSkip) },
		obsBuilder, parser, MakeSetter(args.setter), args.teamSize, true
	);
	Gym gym(match, args.tickSkip);

	int goalsA = 0, goalsB = 0, winsA = 0, winsB = 0, draws = 0;
	int64_t totalSteps = 0;
	auto start = std::chrono::steady_clock::now();

	for (int game = 0; game < args.games; game++) {
		// Seitentausch: in ungeraden Spielen spielt A orange
		bool aIsBlue = (game % 2) == 0;
		// Audit K2: Die Beobachtung kommt aus dem Gym (Reset/Step), genau wie im Training.
		// BuildOBS hier noch einmal aufzurufen würde den Aktions-Stack ein zweites Mal
		// fortschreiben und die Policy sähe jede Aktion doppelt.
		FList2 obsSet = gym.Reset();

		GameState state = gym.prevState;
		int scoreBlue = 0, scoreOrange = 0;
		bool done = false;
		int steps = 0;

		while (!done) {
			IList actions(state.players.size());
			for (size_t pi = 0; pi < state.players.size(); pi++) {
				auto& player = state.players[pi];
				bool usesA = (player.team == Team::BLUE) == aIsBlue;
				auto& policy = usesA ? a : b;

				const FList& obs = obsSet[pi];
				auto input = torch::from_blob(const_cast<float*>(obs.data()), { 1, (int64_t)obs.size() }).clone();
				auto probs = PolicyProbs(policy.seq, input, args.temperature);

				int action;
				if (args.deterministic) {
					action = (int)probs.argmax(1).item<int64_t>();
				} else {
					action = (int)torch::multinomial(probs, 1, true).item<int64_t>();
				}
				actions[pi] = action;
			}

			auto result = gym.Step(actions);
			obsSet = result.obs;
			state = result.state;
			done = result.done;
			steps++;
			totalSteps++;

			scoreBlue = state.scoreLine[(int)Team::BLUE];
			scoreOrange = state.scoreLine[(int)Team::ORANGE];
		}

		int gA = aIsBlue ? scoreBlue : scoreOrange;
		int gB = aIsBlue ? scoreOrange : scoreBlue;
		goalsA += gA;
		goalsB += gB;
		if (gA > gB) winsA++;
		else if (gB > gA) winsB++;
		else draws++;

		if ((game + 1) % 10 == 0 || game + 1 == args.games)
			std::cout << "Spiel " << (game + 1) << "/" << args.games
			          << "  Tore " << goalsA << ":" << goalsB
			          << "  Siege " << winsA << ":" << winsB << " (" << draws << " remis)\n";
	}

	double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

	json out = {
		{ "policy_a", args.pathA }, { "policy_b", args.pathB },
		{ "games", args.games }, { "team_size", args.teamSize },
		{ "deterministic", args.deterministic }, { "temperature", args.temperature },
		{ "goals_a", goalsA }, { "goals_b", goalsB },
		{ "wins_a", winsA }, { "wins_b", winsB }, { "draws", draws },
		{ "steps", totalSteps }, { "seconds", seconds },
	};
	std::ofstream(args.outPath) << out.dump(2);
	std::cout << "Ergebnis in " << args.outPath << " (" << seconds << " s)\n";
	return 0;
}
