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
//
// Aktionsauswahl je Seite (--a-mode, --b-mode): sample (Standard, wie im Training), argmax (wie
// der RLBot-Bot bisher) oder argmax_group (argmax über gleichwirkende Einträge, env/cpp/ActionSelect.h).
// --deterministic setzt beide Seiten auf argmax.
//
// Kennzahlen je Seite ("stats" im JSON, env/cpp/PlayStats.h): Anstöße (erste Berührung, Ballbesitz
// 3 s danach, Tore in 10 s), Aufenthalte im Angriffsdrittel (mit/ohne Tor, lange), Schüsse,
// Ballkontakte am Boden/in der Luft mit Höhe, im Teamspiel Double-Commits und Absicherung.
#include "policy_io.h"

#include "env/cpp/ActionSelect.h"
#include "env/cpp/Obs.h"
#include "env/cpp/PlayStats.h"
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
	SelectMode modeA = SelectMode::SAMPLE, modeB = SelectMode::SAMPLE;
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

// Kennzahlen einer Seite, als Summen über alle Spiele (Raten entstehen erst in der Ausgabe)
struct SideStats {
	int64_t kickoffs = 0, firstTouches = 0, untouched = 0, possessionHalf = 0, closer = 0, kickoffGoals = 0;
	double firstTouchTime = 0, firstTouchSpeed = 0, kickoffBoost = 0;
	int64_t spells = 0, spellGoals = 0, longSpells = 0, thirdSteps = 0;
	double spellSeconds = 0, spellNoGoalSeconds = 0;
	int64_t shots = 0, touches = 0, airTouches = 0, aerialTouches = 0;
	double touchHeight = 0, airTouchHeight = 0;
	int64_t teamSamples = 0, doubleCommits = 0, lastBack = 0;
	double mateDist = 0;

	void Add(const SideStats& o) {
		kickoffs += o.kickoffs; firstTouches += o.firstTouches; untouched += o.untouched;
		possessionHalf += o.possessionHalf; closer += o.closer; kickoffGoals += o.kickoffGoals;
		firstTouchTime += o.firstTouchTime; firstTouchSpeed += o.firstTouchSpeed; kickoffBoost += o.kickoffBoost;
		spells += o.spells; spellGoals += o.spellGoals; longSpells += o.longSpells; thirdSteps += o.thirdSteps;
		spellSeconds += o.spellSeconds; spellNoGoalSeconds += o.spellNoGoalSeconds;
		shots += o.shots; touches += o.touches; airTouches += o.airTouches; aerialTouches += o.aerialTouches;
		touchHeight += o.touchHeight; airTouchHeight += o.airTouchHeight;
		teamSamples += o.teamSamples; doubleCommits += o.doubleCommits; lastBack += o.lastBack; mateDist += o.mateDist;
	}
};

// Lange Aufenthalte im Angriffsdrittel (wie OFF_THIRD_LONG_SECS in train/cpp/Metrics.h)
constexpr float LONG_SPELL_SECS = 8.f;

// Ereignisse eines Schritts auf die Seiten verteilen; side[team] = 0 für A, 1 für B.
static void AddEvents(const PlayEvents& ev, const int side[2], SideStats stats[2]) {
	for (auto& ko : ev.kickoffs)
		for (int s = 0; s < 2; s++) {
			SideStats& st = stats[s];
			st.kickoffs++;
			st.kickoffBoost += ko.boostUsed[side[0] == s ? 0 : 1];
			if (ko.firstTeam < 0) {
				st.untouched++;
				continue;
			}
			if (side[ko.firstTeam] == s) {
				st.firstTouches++;
				st.firstTouchTime += ko.timeToTouch;
				st.firstTouchSpeed += ko.touchSpeed;
			}
			st.possessionHalf += ko.ballHalfTeam >= 0 && side[ko.ballHalfTeam] == s;
			st.closer += ko.closerTeam >= 0 && side[ko.closerTeam] == s;
			st.kickoffGoals += ko.goalTeam >= 0 && side[ko.goalTeam] == s;
		}
	for (auto& sp : ev.spells) {
		SideStats& st = stats[side[sp.team]];
		st.spells++;
		st.spellSeconds += sp.seconds;
		st.spellGoals += sp.goal;
		st.longSpells += sp.seconds >= LONG_SPELL_SECS;
		if (!sp.goal)
			st.spellNoGoalSeconds += sp.seconds;
	}
	for (auto& t : ev.touches) {
		SideStats& st = stats[side[t.team]];
		st.touches++;
		st.touchHeight += t.ballHeight;
		if (t.carInAir) {
			st.airTouches++;
			st.airTouchHeight += t.ballHeight;
			st.aerialTouches += t.ballHeight >= AERIAL_TOUCH_MIN_HEIGHT;
		}
	}
	for (auto& ts : ev.teams) {
		SideStats& st = stats[side[ts.team]];
		st.teamSamples++;
		st.doubleCommits += ts.doubleCommit;
		st.lastBack += ts.lastBack;
		st.mateDist += ts.mateDist;
	}
	for (int team = 0; team < 2; team++)
		stats[side[team]].shots += ev.shots[team];
	if (ev.ballThirdTeam >= 0)
		stats[side[ev.ballThirdTeam]].thirdSteps++;
}

static json StatsJSON(const SideStats& s, double gameMinutes, int playersPerSide, int tickSkip) {
	auto div = [](double a, double b) { return b > 0 ? a / b : 0.0; };
	double playerMinutes = gameMinutes * playersPerSide;
	double stepsPerMin = 60.0 * 120.0 / tickSkip;
	json j = {
		{ "kickoffs", s.kickoffs }, { "kickoff_first_touches", s.firstTouches },
		{ "kickoff_untouched", s.untouched },
		{ "kickoff_first_touch_rate", div(s.firstTouches, s.kickoffs - s.untouched) },
		{ "kickoff_ball_half_rate", div(s.possessionHalf, s.kickoffs - s.untouched) },
		{ "kickoff_closer_rate", div(s.closer, s.kickoffs - s.untouched) },
		{ "kickoff_goals_10s", s.kickoffGoals }, { "kickoff_goal_rate", div(s.kickoffGoals, s.kickoffs) },
		{ "kickoff_first_touch_s", div(s.firstTouchTime, s.firstTouches) },
		{ "kickoff_touch_speed", div(s.firstTouchSpeed, s.firstTouches) },
		{ "kickoff_boost_used", div(s.kickoffBoost, s.kickoffs) },
		{ "off_third_spells", s.spells }, { "off_third_goals", s.spellGoals },
		{ "off_third_conversion", div(s.spellGoals, s.spells) },
		{ "off_third_long", s.longSpells }, { "off_third_long_share", div(s.longSpells, s.spells) },
		{ "off_third_share", div(s.thirdSteps, gameMinutes * stepsPerMin) },
		{ "off_third_nogoal_s_per_min", div(s.spellNoGoalSeconds, gameMinutes) },
		{ "shots", s.shots }, { "shots_per_min", div(s.shots, playerMinutes) },
		{ "touches", s.touches }, { "touches_per_min", div(s.touches, playerMinutes) },
		{ "air_touches", s.airTouches }, { "air_touch_per_min", div(s.airTouches, playerMinutes) },
		{ "aerial_touches", s.aerialTouches }, { "aerial_touch_per_min", div(s.aerialTouches, playerMinutes) },
		{ "touch_height_mean", div(s.touchHeight, s.touches) },
		{ "air_touch_height_mean", div(s.airTouchHeight, s.airTouches) },
	};
	if (s.teamSamples > 0) {
		j["double_commit"] = div(s.doubleCommits, s.teamSamples);
		j["last_back"] = div(s.lastBack, s.teamSamples);
		j["mate_dist"] = div(s.mateDist, s.teamSamples);
	}
	return j;
}

struct GameResult {
	bool aIsBlue = true;
	int goalsA = 0, goalsB = 0, kickoffs = 0, firstSeed = 0;
	double gameSeconds = 0, wallSeconds = 0;
	json goals = json::array();
	uint64_t fingerprint = 0;
	int64_t steps = 0;
	SideStats stats[2];   // [0] = A, [1] = B
};

// Wirkungsklassen der Aktionstabelle am Boden und in der Luft (für argmax_group)
static const std::vector<int>& Groups(bool onGround) {
	static const std::vector<int> ground = EffectGroups(DiscreteAction().actions, true);
	static const std::vector<int> air = EffectGroups(DiscreteAction().actions, false);
	return onGround ? ground : air;
}

static GameResult PlayGame(const Args& args, int game, torch::nn::Sequential& seqA, torch::nn::Sequential& seqB) {
	GameResult res;
	res.aIsBlue = (game % 2) == 0;
	const int side[2] = { res.aIsBlue ? 0 : 1, res.aIsBlue ? 1 : 0 };   // Team -> Seite (0 = A)
	PlayTracker tracker(args.tickSkip);
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
			SelectMode mode = usesA ? args.modeA : args.modeB;
			int action;
			if (mode == SelectMode::ARGMAX)
				action = (int)probs.argmax(1).item<int64_t>();
			else if (mode == SelectMode::ARGMAX_GROUP)
				action = ArgmaxGroup(probs.contiguous().data_ptr<float>(), Groups(player.carState.isOnGround));
			else
				action = (int)torch::multinomial(probs, 1, true, gen).item<int64_t>();
			actions[pi] = action;
		}

		auto result = gym->Step(actions);
		state = result.state;
		res.steps++;
		bool goal = result.done && RLGSC::Math::IsBallScored(state.ball.pos);
		AddEvents(tracker.Step(state, goal), side, res.stats);

		if (goal) {
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
	AddEvents(tracker.Flush(), side, res.stats);
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
		else if (arg == "--a-mode" || arg == "--b-mode") {
			std::string v = next();
			if (!ParseSelectMode(v, arg == "--a-mode" ? args.modeA : args.modeB)) {
				std::cerr << "Unbekannte Aktionsauswahl: " << v << " (sample, argmax, argmax_group)\n";
				return 2;
			}
		}
		else if (arg == "--allow-duplicates") args.allowDuplicates = true;
		else if (arg == "--temperature") args.temperature = std::stof(next());
		else if (arg == "--seed") args.seed = std::stoi(next());
		else if (arg == "--setter") args.setter = next();
		else { std::cerr << "Unbekanntes Argument: " << arg << "\n"; return 2; }
	}
	if (args.pathA.empty() || args.pathB.empty()) {
		std::cerr << "usage: duel --a <policy.lt> --b <policy.lt> [--games N] [--max-seconds 300] [--seed N] "
		             "[--threads N] [--team-size N] [--a-mode|--b-mode sample|argmax|argmax_group] "
		             "[--deterministic [--allow-duplicates]] [--out result.json]\n";
		return 2;
	}
	if (args.deterministic)
		args.modeA = args.modeB = SelectMode::ARGMAX;
	// Deterministische Policies + Anstoß = identische Spiele, sobald sich Anstoßposition und Seite
	// wiederholen. Mehr Spiele als verschiedene Starts wären keine unabhängigen Messungen. Das gilt,
	// sobald keine Seite zieht (argmax und argmax_group sind beide deterministisch).
	bool bothDeterministic = args.modeA != SelectMode::SAMPLE && args.modeB != SelectMode::SAMPLE;
	int distinctStarts = 2 * KICKOFF_VARIANTS;
	if (bothDeterministic && args.setter == "kickoff" && args.games > distinctStarts && !args.allowDuplicates) {
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
	SideStats statsA, statsB;
	double gameMinutes = 0;
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
		statsA.Add(r.stats[0]);
		statsB.Add(r.stats[1]);
		gameMinutes += r.gameSeconds / 60.0;
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
		{ "deterministic", bothDeterministic }, { "temperature", args.temperature },
		{ "mode_a", SelectModeName(args.modeA) }, { "mode_b", SelectModeName(args.modeB) },
		{ "goals_a", goalsA }, { "goals_b", goalsB },
		{ "wins_a", winsA }, { "wins_b", winsB }, { "draws", draws },
		{ "distinct_games", distinct },
		{ "steps", totalSteps }, { "seconds", seconds },
		{ "cpu_seconds_per_game", args.games ? wallSum / args.games : 0.0 },
		{ "per_game", perGame },
		{ "stats", { { "a", StatsJSON(statsA, gameMinutes, args.teamSize, args.tickSkip) },
		             { "b", StatsJSON(statsB, gameMinutes, args.teamSize, args.tickSkip) } } },
	};
	std::ofstream(args.outPath) << out.dump(2);
	std::cout << "Ergebnis in " << args.outPath << " (" << seconds << " s, " << threads << " Threads)\n";
	return 0;
}
