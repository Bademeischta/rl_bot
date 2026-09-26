// Lässt zwei Checkpoints in RocketSim gegeneinander spielen und schreibt das Ergebnis als JSON.
//
//   duel.exe --a <PPO_POLICY.lt> --b <PPO_POLICY.lt> --games 1000 --out result.json [--threads N]
//
// Ein Spiel ist ein Match über --max-seconds Spielzeit (Standard 300 s wie in Rocket League):
// Nach einem Tor geht es mit einem neuen Anstoß weiter, bis die Zeit um ist. Gezählt werden die
// Tore je Spiel; Hauptkennzahl der Auswertung ist die Tordifferenz pro Spiel (compare.py).
// Vorher endete ein Spiel beim ersten Tor oder nach 120 s; im Probelauf waren so 17 von 20 Spielen
// torlos und die Tordifferenz je Spiel nur -1/0/+1.
//
// Unabhängigkeit und Reproduzierbarkeit der Spiele:
//  - Seitentausch: in ungeraden Spielen spielt A orange. Die Spiele 2k und 2k+1 bilden ein Paar mit
//    derselben Anstoß-Folge (gleiche Seeds), nur die Seiten sind vertauscht; so heben sich Vorteile
//    einzelner Anstoßpositionen im Paar auf.
//  - Anstöße sind geseedet (--seed, Paar, Anstoß-Nummer). RocketSims ResetToRandomKickoff zog
//    vorher aus einem zeitgeseedeten Zufallsgenerator; Duelle waren nicht reproduzierbar.
//  - Jedes Spiel bekommt eine frische Arena und eigene, aus (--seed, Spiel) geseedete
//    Zufallsgeneratoren (Aktionen; RocketSims Respawn nach Demolierung). Ein Spiel hängt damit nicht
//    von vorherigen Spielen ab (Bullet nimmt Kontakt-Zustand über einen Reset mit).
//  - Bitgenau reproduzierbar sind Spiele trotzdem nicht: RocketSim iteriert Autos in einem
//    unordered_set (adressabhängig), die Physik weicht zwischen zwei Läufen in Nachkommastellen ab
//    und das kippt gelegentlich eine gezogene Aktion. Gemessen: 31 von 32 Spielen mit gleichen
//    Toren zwischen zwei Läufen mit gleichem Seed. Für die Statistik ist das zusätzlicher Zufall,
//    die Spiele bleiben unabhängig.
//  - Die Aktionen werden aus der Policy gezogen (Standard). --deterministic nimmt die
//    wahrscheinlichste Aktion; dann sind zwei Spiele mit gleicher Anstoßposition und Seite
//    identisch (nur 5 Anstoßpositionen). duel.exe zählt verschiedene Spiele über einen
//    Fingerabdruck ("distinct_games") und lehnt --deterministic für mehr Spiele ab, als es
//    verschiedene Starts gibt, außer mit --allow-duplicates.
//  - Spiele laufen parallel (--threads, Standard: halbe Anzahl logischer Kerne = physische Kerne
//    auf dem Ryzen 7 8700F). Gemessen: 1,14 s je Spiel à 300 s auf einem Kern; 8 Threads 0,35 s
//    je Spiel (Faktor 3,2), 16 Threads nicht schneller.
#include "policy_io.h"

#include "env/cpp/Obs.h"
#include "env/cpp/StateSetters.h"

#include <RLGymSim_CPP/Gym.h>
#include <RLGymSim_CPP/Math.h>
#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/StateSetters/RandomState.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>

#include <ATen/CPUGeneratorImpl.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <thread>

using namespace RLGSC;
using namespace RLbot;
using nlohmann::json;

// In 1v1 gibt RocketSim 5 Anstoßpositionen vor (Soccar), je Seite gespiegelt.
constexpr int KICKOFF_VARIANTS = 5;

struct Args {
	std::string pathA, pathB, outPath = "duel_result.json";
	std::string meshDir = "collision_meshes";
	int games = 50, teamSize = 1, tickSkip = 8, maxSeconds = 300, actionStack = 5, maxPlayers = 3;
	int threads = 0;   // 0 = physische Kerne (logische / 2)
	bool deterministic = false, allowDuplicates = false;
	float temperature = 1.f;
	int seed = 123;
	// kickoff ist der Standard für vergleichbare Ratings; defense/random dienen dazu,
	// die Torzählung zu prüfen und gezielte Szenen zu bewerten.
	std::string setter = "kickoff";
};

// Anstoß mit festem Seed: RocketSims ResetToRandomKickoff(seed) mischt die Positionen mit einem
// eigenen Generator statt mit dem zeitgeseedeten globalen.
class SeededKickoffSetter : public StateSetter {
public:
	int nextSeed = 0;
	virtual GameState ResetState(Arena* arena) {
		arena->ResetToRandomKickoff(nextSeed);
		return GameState(arena);
	}
};

static StateSetter* MakeSetter(const std::string& name) {
	if (name == "kickoff") return new SeededKickoffSetter();
	if (name == "defense") return new DefenseSetter();
	if (name == "random") return new RandomState(true, true, true);
	if (name == "aerial") return new AerialSetter();
	std::cerr << "Unbekannter State-Setter: " << name << "\n";
	exit(2);
}

// Seed des n-ten Anstoßes im Spielpaar (beide Spiele eines Paars bekommen dieselben Seeds).
static int KickoffSeed(int base, int pair, int kickoff) {
	uint64_t s = (uint64_t)(uint32_t)base * 1000003ull + (uint64_t)pair * 7919ull + (uint64_t)kickoff * 104729ull;
	return (int)(s % 2147483647ull);
}

// Seed des Aktions-Zufallsgenerators eines Spiels.
static uint64_t ActionSeed(int base, int game) {
	return (uint64_t)(uint32_t)base * 2654435761ull + (uint64_t)game * 40503ull + 1;
}

static uint64_t Fnv(uint64_t h, int64_t v) {
	for (int i = 0; i < 8; i++) {
		h ^= (uint64_t)((v >> (8 * i)) & 0xff);
		h *= 1099511628211ull;
	}
	return h;
}

struct GameResult {
	bool aIsBlue = true;
	int goalsA = 0, goalsB = 0, kickoffs = 0, firstSeed = 0;
	double gameSeconds = 0, wallSeconds = 0;
	json goals = json::array();
	uint64_t fingerprint = 0;
	int64_t steps = 0;
};

static GameResult PlayGame(const Args& args, int game, torch::nn::Sequential& seqA, torch::nn::Sequential& seqB) {
	GameResult res;
	res.aIsBlue = (game % 2) == 0;
	int pair = game / 2;
	auto wallStart = std::chrono::steady_clock::now();

	StateSetter* setter = MakeSetter(args.setter);
	auto* seededKickoff = dynamic_cast<SeededKickoffSetter*>(setter);
	// Nur das Tor beendet einen Abschnitt; die Spielzeit zählt die Schleife selbst.
	auto* match = new Match(new EventReward({}), { new GoalScoreCondition() },
	                        new StackedPaddedOBS(args.maxPlayers, args.actionStack, false),
	                        new DiscreteAction(), setter, args.teamSize, true);
	auto* gym = new Gym(match, args.tickSkip);
	at::Generator gen = at::detail::createCPUGenerator(ActionSeed(args.seed, game));
	// RocketSim zieht die Respawn-Position nach einer Demolierung (Car::Respawn(-1)) aus seinem
	// thread-lokalen, zeitgeseedeten Generator. Das Spiel läuft komplett in diesem Thread, also
	// den Generator hier je Spiel neu seeden; sonst weichen Spiele mit Demos zufällig ab.
	RocketSim::Math::GetRandEngine().seed((unsigned)(ActionSeed(args.seed, game) ^ 0x5bd1e995u));

	const int stepsPerGame = args.maxSeconds * 120 / args.tickSkip;
	int kickoff = 0;
	res.firstSeed = KickoffSeed(args.seed, pair, 0);
	if (seededKickoff)
		seededKickoff->nextSeed = res.firstSeed;
	// Audit K2: Die Beobachtung kommt aus dem Gym (Reset/Step), genau wie im Training.
	FList2 obsSet = gym->Reset();
	GameState state = gym->prevState;
	uint64_t fp = 1469598103934665603ull;

	for (int step = 0; step < stepsPerGame; step++) {
		IList actions(state.players.size());
		// Aktionen in fester Reihenfolge (Car-ID) ziehen: state.players folgt arena->_cars, einem
		// unordered_set mit adressabhängiger Reihenfolge (vgl. R19). Sonst bekäme je nach Lauf
		// mal Blau, mal Orange die erste Zufallszahl und gleiche Seeds ergäben andere Spiele.
		std::vector<size_t> order(state.players.size());
		for (size_t pi = 0; pi < order.size(); pi++) order[pi] = pi;
		std::sort(order.begin(), order.end(),
		          [&](size_t x, size_t y) { return state.players[x].carId < state.players[y].carId; });
		for (size_t pi : order) {
			auto& player = state.players[pi];
			bool usesA = (player.team == Team::BLUE) == res.aIsBlue;
			auto& seq = usesA ? seqA : seqB;

			const FList& obs = obsSet[pi];
			auto input = torch::from_blob(const_cast<float*>(obs.data()), { 1, (int64_t)obs.size() }).clone();
			auto probs = PolicyProbs(seq, input, args.temperature);
			int action;
			if (args.deterministic)
				action = (int)probs.argmax(1).item<int64_t>();
			else
				action = (int)torch::multinomial(probs, 1, true, gen).item<int64_t>();
			actions[pi] = action;
		}

		auto result = gym->Step(actions);
		state = result.state;
		res.steps++;

		if (result.done && RLGSC::Math::IsBallScored(state.ball.pos)) {
			// Ball im Tor mit y > 0 (oranges Tor) = Tor für Blau
			bool blueScored = state.ball.pos.y > 0;
			bool aScored = blueScored == res.aIsBlue;
			(aScored ? res.goalsA : res.goalsB)++;
			res.goals.push_back({ { "step", step + 1 }, { "by", aScored ? "a" : "b" } });
			fp = Fnv(Fnv(fp, step + 1), aScored ? 1 : 2);
			// Weiter mit einem neuen Anstoß (wie nach einem Tor im Spiel, ohne Replay)
			kickoff++;
			if (seededKickoff)
				seededKickoff->nextSeed = KickoffSeed(args.seed, pair, kickoff);
			obsSet = gym->Reset();
			state = gym->prevState;
		} else {
			obsSet = result.obs;
		}
	}
	// Endzustand in den Fingerabdruck: zwei Spiele mit gleichem Abdruck sind Wiederholungen
	fp = Fnv(fp, (int64_t)std::llround(state.ball.pos.x * 10));
	fp = Fnv(fp, (int64_t)std::llround(state.ball.pos.y * 10));
	fp = Fnv(fp, (int64_t)std::llround(state.ball.pos.z * 10));
	for (auto& p : state.players)
		fp = Fnv(Fnv(fp, (int64_t)std::llround(p.phys.pos.x * 10)), (int64_t)std::llround(p.phys.pos.y * 10));

	delete gym;
	delete match;
	res.kickoffs = kickoff + 1;
	res.fingerprint = fp;
	res.gameSeconds = stepsPerGame * args.tickSkip / 120.0;
	res.wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
	return res;
}

static torch::nn::Sequential CloneSeq(const torch::nn::Sequential& seq) {
	return torch::nn::Sequential(std::dynamic_pointer_cast<torch::nn::SequentialImpl>(seq->clone()));
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
		else if (arg == "--threads") args.threads = std::stoi(next());
		else if (arg == "--deterministic") args.deterministic = true;
		else if (arg == "--allow-duplicates") args.allowDuplicates = true;
		else if (arg == "--temperature") args.temperature = std::stof(next());
		else if (arg == "--seed") args.seed = std::stoi(next());
		else if (arg == "--setter") args.setter = next();
		else { std::cerr << "Unbekanntes Argument: " << arg << "\n"; return 2; }
	}
	if (args.pathA.empty() || args.pathB.empty()) {
		std::cerr << "usage: duel --a <policy.lt> --b <policy.lt> [--games N] [--max-seconds 300] [--seed N] "
		             "[--threads N] [--team-size N] [--deterministic [--allow-duplicates]] [--out result.json]\n";
		return 2;
	}
	// Deterministische Policies + Anstoß = identische Spiele, sobald sich Anstoßposition und Seite
	// wiederholen. Mehr Spiele als verschiedene Starts wären keine unabhängigen Messungen.
	int distinctStarts = 2 * KICKOFF_VARIANTS;
	if (args.deterministic && args.setter == "kickoff" && args.games > distinctStarts && !args.allowDuplicates) {
		std::cerr << "--deterministic mit Anstoß erlaubt höchstens " << distinctStarts << " verschiedene Spiele "
		          << "(5 Anstoßpositionen x 2 Seiten); " << args.games << " Spiele wären Wiederholungen. "
		          << "Ohne --deterministic spielen (Standard) oder --allow-duplicates setzen.\n";
		return 2;
	}

	torch::NoGradGuard noGrad;
	// Winzige Einzel-Inferenzen: Parallelität über Spiele statt innerhalb von torch
	torch::set_num_threads(1);

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

	int threads = args.threads > 0 ? args.threads : (int)std::max(1u, std::thread::hardware_concurrency() / 2);
	threads = std::min(threads, std::max(1, args.games));
	std::vector<GameResult> results(args.games);
	std::atomic<int> nextGame = 0, finished = 0;
	std::mutex printMutex;
	auto start = std::chrono::steady_clock::now();

	auto worker = [&]() {
		torch::NoGradGuard workerNoGrad;   // Grad-Modus ist thread-lokal
		torch::set_num_threads(1);         // die OpenMP-Threadzahl ebenfalls
		// Eigene Kopien der Netze je Thread
		auto seqA = CloneSeq(a.seq), seqB = CloneSeq(b.seq);
		for (int game = nextGame++; game < args.games; game = nextGame++) {
			results[game] = PlayGame(args, game, seqA, seqB);
			int done = ++finished;
			if (done % 50 == 0 || done == args.games) {
				std::lock_guard<std::mutex> lock(printMutex);
				std::cout << "Spiele fertig: " << done << "/" << args.games << "\n";
			}
		}
	};
	std::vector<std::thread> pool;
	for (int t = 0; t < threads; t++)
		pool.emplace_back(worker);
	for (auto& t : pool)
		t.join();

	double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

	int goalsA = 0, goalsB = 0, winsA = 0, winsB = 0, draws = 0;
	int64_t totalSteps = 0;
	double wallSum = 0;
	json perGame = json::array();
	std::set<uint64_t> fingerprints;
	for (auto& r : results) {
		goalsA += r.goalsA;
		goalsB += r.goalsB;
		if (r.goalsA > r.goalsB) winsA++;
		else if (r.goalsB > r.goalsA) winsB++;
		else draws++;
		totalSteps += r.steps;
		wallSum += r.wallSeconds;
		fingerprints.insert(r.fingerprint);
		perGame.push_back({
			{ "a_blue", r.aIsBlue }, { "goals_a", r.goalsA }, { "goals_b", r.goalsB },
			{ "game_seconds", r.gameSeconds }, { "wall_seconds", r.wallSeconds },
			{ "kickoffs", r.kickoffs }, { "first_kickoff_seed", r.firstSeed },
			{ "goals", r.goals }, { "fingerprint", std::to_string(r.fingerprint) },
		});
	}
	int distinct = (int)fingerprints.size();
	std::cout << "Tore " << goalsA << ":" << goalsB << "  Siege " << winsA << ":" << winsB
	          << " (" << draws << " remis)\n";
	if (distinct < args.games)
		std::cout << "WARNUNG: nur " << distinct << " verschiedene von " << args.games
		          << " Spielen (Wiederholungen, keine unabhängigen Messungen)\n";

	json out = {
		{ "policy_a", args.pathA }, { "policy_b", args.pathB },
		{ "games", args.games }, { "team_size", args.teamSize }, { "max_seconds", args.maxSeconds },
		{ "seed", args.seed }, { "setter", args.setter }, { "threads", threads },
		{ "deterministic", args.deterministic }, { "temperature", args.temperature },
		{ "goals_a", goalsA }, { "goals_b", goalsB },
		{ "wins_a", winsA }, { "wins_b", winsB }, { "draws", draws },
		{ "distinct_games", distinct },
		{ "steps", totalSteps }, { "seconds", seconds },
		{ "cpu_seconds_per_game", args.games ? wallSum / args.games : 0.0 },
		{ "per_game", perGame },
	};
	std::ofstream(args.outPath) << out.dump(2);
	std::cout << "Ergebnis in " << args.outPath << " (" << seconds << " s, " << threads << " Threads)\n";
	return 0;
}
