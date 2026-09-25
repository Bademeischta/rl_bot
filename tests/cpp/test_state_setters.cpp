// Prüft die State-Setter des Mechanik-Curriculums an einer echten RocketSim-Arena.
// Braucht die Collision-Meshes; ohne sie werden diese Tests übersprungen.
#include "test_util.h"

#include "env/cpp/StateSetters.h"

#include <map>
#include <set>

using namespace RLGSC;
using namespace RLbot;

extern bool g_arenaReady;

static Arena* MakeArena(int teamSize) {
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < teamSize; i++) arena->AddCar(Team::BLUE);
	for (int i = 0; i < teamSize; i++) arena->AddCar(Team::ORANGE);
	return arena;
}

static bool IsFinite(Vec v) {
	return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// Grundprüfung, die für jede Szene gelten muss: alles endlich, alles im Feld,
// Autos über dem Boden, Ball nicht in der Wand.
static void CheckStateSane(const GameState& state, const char* what) {
	using namespace CommonValues;
	if (!IsFinite(state.ball.pos)) FAIL_AT(what << ": Ballposition nicht endlich");
	if (!IsFinite(state.ball.vel)) FAIL_AT(what << ": Ballgeschwindigkeit nicht endlich");
	if (state.ball.pos.z < BALL_RADIUS - 1.f) FAIL_AT(what << ": Ball unter dem Boden");
	if (std::abs(state.ball.pos.x) > SIDE_WALL_X) FAIL_AT(what << ": Ball außerhalb in X");
	if (std::abs(state.ball.pos.y) > BACK_NET_Y) FAIL_AT(what << ": Ball außerhalb in Y");
	if (state.ball.pos.z > CEILING_Z) FAIL_AT(what << ": Ball über der Decke");

	for (auto& p : state.players) {
		if (!IsFinite(p.phys.pos)) FAIL_AT(what << ": Autoposition nicht endlich");
		if (!IsFinite(p.phys.vel)) FAIL_AT(what << ": Autogeschwindigkeit nicht endlich");
		if (p.phys.pos.z < 0) FAIL_AT(what << ": Auto unter dem Boden");
		if (p.phys.pos.z > CEILING_Z) FAIL_AT(what << ": Auto über der Decke");
		if (std::abs(p.phys.pos.x) > SIDE_WALL_X) FAIL_AT(what << ": Auto außerhalb in X");
		if (std::abs(p.phys.pos.y) > BACK_NET_Y) FAIL_AT(what << ": Auto außerhalb in Y");
		if (p.boostFraction < 0 || p.boostFraction > 1) FAIL_AT(what << ": Boost außerhalb 0..1");
	}
}

TEST(StateSetter_alle_Szenen_liefern_gueltige_Zustaende) {
	if (!g_arenaReady) return;

	struct Entry { const char* name; StateSetter* setter; };
	std::vector<Entry> entries = {
		{ "kickoff", new KickoffSetter() },
		{ "aerial", new AerialSetter() },
		{ "dribble", new DribbleSetter() },
		{ "wall_play", new WallPlaySetter() },
		{ "recovery", new RecoverySetter() },
		{ "defense", new DefenseSetter() },
	};

	for (int teamSize = 1; teamSize <= 3; teamSize++) {
		Arena* arena = MakeArena(teamSize);
		for (auto& entry : entries) {
			for (int i = 0; i < 25; i++) {
				GameState state = entry.setter->ResetState(arena);
				CHECK_EQ((int)state.players.size(), teamSize * 2);
				CheckStateSane(state, entry.name);
			}
		}
		delete arena;
	}
	for (auto& entry : entries) delete entry.setter;
}

TEST(StateSetter_Szenen_erfuellen_ihre_Absicht) {
	if (!g_arenaReady) return;
	Arena* arena = MakeArena(1);

	{ // Aerial: Ball deutlich über dem Boden
		AerialSetter s;
		for (int i = 0; i < 20; i++) {
			auto state = s.ResetState(arena);
			CHECK_GT(state.ball.pos.z, 600);
		}
	}
	{ // Dribble: Ball fast am Boden, ein Auto sehr nah dran
		DribbleSetter s;
		for (int i = 0; i < 20; i++) {
			auto state = s.ResetState(arena);
			CHECK(state.ball.pos.z < CommonValues::BALL_RADIUS + 60);
			float minDist = 1e9f;
			for (auto& p : state.players)
				minDist = RS_MIN(minDist, (state.ball.pos - p.phys.pos).Length());
			CHECK(minDist < 200);
		}
	}
	{ // Wall-Play: Ball nah an einer Seitenwand und über dem Boden
		WallPlaySetter s;
		for (int i = 0; i < 20; i++) {
			auto state = s.ResetState(arena);
			CHECK_GT(std::abs(state.ball.pos.x), 3300);
			CHECK_GT(state.ball.pos.z, 250);
		}
	}
	{ // Recovery: Autos in der Luft mit Drehrate
		RecoverySetter s;
		for (int i = 0; i < 20; i++) {
			auto state = s.ResetState(arena);
			for (auto& p : state.players) {
				CHECK_GT(p.phys.pos.z, 200);
				CHECK_GT(p.phys.angVel.Length(), 0.1f);
			}
		}
	}
	{ // Defense: Ball fliegt Richtung blaues Tor
		DefenseSetter s;
		for (int i = 0; i < 20; i++) {
			auto state = s.ResetState(arena);
			CHECK(state.ball.vel.y < -300);
		}
	}
	{ // Kickoff: Ball ruht in der Mitte
		KickoffSetter s;
		for (int i = 0; i < 10; i++) {
			auto state = s.ResetState(arena);
			CHECK_NEAR(state.ball.pos.x, 0.0, 1.0);
			CHECK_NEAR(state.ball.pos.y, 0.0, 1.0);
			CHECK_NEAR(state.ball.vel.Length(), 0.0, 1.0);
		}
	}
	delete arena;
}

TEST(WeightedStateSetter_trifft_die_Gewichte) {
	if (!g_arenaReady) return;
	Arena* arena = MakeArena(1);

	StateSetterWeights w = {};
	w.kickoff = 3.f;
	w.random = 1.f;
	w.aerial = 0.f;   // ausgeschaltet -> darf nie vorkommen
	WeightedStateSetter setter(w);

	CHECK_EQ((int)setter.setters.size(), 2);
	CHECK_EQ(setter.names[0], std::string("kickoff"));

	std::map<int, int> counts;
	const int N = 2000;
	for (int i = 0; i < N; i++) {
		setter.ResetState(arena);
		counts[setter.lastPicked]++;
	}
	// 3:1 erwartet, Toleranz für die Zufallsstreuung
	CHECK_NEAR(counts[0] / (double)N, 0.75, 0.05);
	CHECK_NEAR(counts[1] / (double)N, 0.25, 0.05);
	delete arena;
}

TEST(WeightedStateSetter_mit_nur_einer_Szene) {
	if (!g_arenaReady) return;
	Arena* arena = MakeArena(1);

	StateSetterWeights w = {};
	w.kickoff = 0; w.random = 0; w.aerial = 1.f;
	WeightedStateSetter setter(w);
	CHECK_EQ((int)setter.setters.size(), 1);

	for (int i = 0; i < 20; i++) {
		auto state = setter.ResetState(arena);
		CHECK_EQ(setter.lastPicked, 0);
		CHECK_GT(state.ball.pos.z, 600);
	}
	delete arena;
}

// --- Seeds (Audit H6) ----------------------------------------------------

static std::vector<Vec> BallPositions(WeightedStateSetter& setter, Arena* arena, int n) {
	std::vector<Vec> out;
	for (int i = 0; i < n; i++)
		out.push_back(setter.ResetState(arena).ball.pos);
	return out;
}

TEST(Seed_gleicher_Seed_gibt_gleiche_Szenenfolge_und_Zustaende) {
	if (!g_arenaReady) return;
	Arena* arena = MakeArena(1);

	// Nur die eigenen Szenen: kickoff und random ziehen aus RocketSims globalem Engine
	StateSetterWeights w = {};
	w.kickoff = 0; w.random = 0;
	w.aerial = 1; w.dribble = 1; w.wallPlay = 1; w.recovery = 1; w.defense = 1;

	WeightedStateSetter a(w, 4711), b(w, 4711);
	CHECK_EQ(a.seed, (int64_t)4711);
	for (int i = 0; i < 40; i++) {
		auto sa = a.ResetState(arena);
		auto sb = b.ResetState(arena);
		CHECK_EQ(a.lastPicked, b.lastPicked);
		CHECK_NEAR(sa.ball.pos.x, sb.ball.pos.x, 1e-6);
		CHECK_NEAR(sa.ball.pos.y, sb.ball.pos.y, 1e-6);
		CHECK_NEAR(sa.ball.pos.z, sb.ball.pos.z, 1e-6);
		CHECK_NEAR(sa.ball.vel.x, sb.ball.vel.x, 1e-6);
		for (size_t p = 0; p < sa.players.size(); p++) {
			CHECK_NEAR(sa.players[p].phys.pos.x, sb.players[p].phys.pos.x, 1e-6);
			CHECK_NEAR(sa.players[p].phys.pos.y, sb.players[p].phys.pos.y, 1e-6);
			CHECK_NEAR(sa.players[p].boostFraction, sb.players[p].boostFraction, 1e-6);
		}
	}
	delete arena;
}

TEST(Seed_verschiedene_Seeds_geben_verschiedene_Zustaende) {
	if (!g_arenaReady) return;
	Arena* arena = MakeArena(1);
	StateSetterWeights w = {};
	w.kickoff = 0; w.random = 0; w.aerial = 1;

	WeightedStateSetter a(w, 1), b(w, 2);
	auto pa = BallPositions(a, arena, 10);
	auto pb = BallPositions(b, arena, 10);
	int different = 0;
	for (int i = 0; i < 10; i++)
		different += std::abs(pa[i].x - pb[i].x) > 1e-3;
	CHECK_GT(different, 8);
	delete arena;
}

TEST(Seed_ungeseedet_bleibt_alter_Pfad) {
	if (!g_arenaReady) return;
	Arena* arena = MakeArena(1);
	StateSetterWeights w = {};
	w.kickoff = 0; w.random = 0; w.aerial = 1;

	WeightedStateSetter a(w), b(w);
	CHECK_EQ(a.seed, (int64_t)-1);
	auto pa = BallPositions(a, arena, 10);
	auto pb = BallPositions(b, arena, 10);
	int different = 0;
	for (int i = 0; i < 10; i++)
		different += std::abs(pa[i].x - pb[i].x) > 1e-3;
	CHECK_GT(different, 8);   // globaler Engine: zwei Setter laufen nicht synchron
	delete arena;
}

TEST(Seed_Mischung_trennt_benachbarte_Envs_und_Stroeme) {
	std::set<uint64_t> seen;
	for (int env = 0; env < 64; env++)
		for (int stream = 0; stream < 2; stream++)
			seen.insert(SeedForEnv(123, env, stream));
	CHECK_EQ((int)seen.size(), 128);
	CHECK(SeedForEnv(123, 0, 0) != SeedForEnv(124, 0, 0));
	CHECK_EQ(SeedForEnv(123, 5, 1), SeedForEnv(123, 5, 1));
}

// Nebenbefund R19 zu H6: arena->_cars ist ein std::unordered_set<Car*>, die Iterationsreihenfolge
// hängt von den Speicheradressen ab. Gleicher Seed muss trotzdem jedem Auto (per Car-ID) denselben
// Zustand geben, sonst sind geseedete Läufe im 1v1 nicht reproduzierbar.
TEST(Seed_gleicher_Seed_gibt_gleiche_Autozustaende_unabhaengig_von_der_Speicherreihenfolge) {
	if (!g_arenaReady) return;
	StateSetterWeights w = {};
	w.kickoff = 0; w.random = 0;
	w.aerial = 1; w.dribble = 1; w.wallPlay = 1; w.recovery = 1; w.defense = 1;

	// Viele Arenen mit Füllallokationen dazwischen, damit die Autos an verschiedenen Adressen
	// landen und beide Iterationsreihenfolgen vorkommen.
	std::vector<Arena*> arenas;
	std::vector<std::vector<char>> padding;
	int lowIdFirst = 0, highIdFirst = 0;
	for (int i = 0; i < 40; i++) {
		padding.emplace_back(64 + 97 * i);
		Arena* a = Arena::Create(GameMode::SOCCAR);
		a->AddCar(Team::BLUE);
		padding.emplace_back(32 + 211 * i);
		a->AddCar(Team::ORANGE);
		uint32_t minId = UINT32_MAX;
		for (Car* c : a->_cars) minId = RS_MIN(minId, c->id);
		((*a->_cars.begin())->id == minId ? lowIdFirst : highIdFirst)++;
		arenas.push_back(a);
	}
	// Voraussetzung des Tests: beide Reihenfolgen kommen tatsächlich vor
	CHECK_GT(lowIdFirst, 0);
	CHECK_GT(highIdFirst, 0);

	auto carStates = [&](Arena* arena) {
		WeightedStateSetter setter(w, 4711);
		std::vector<std::map<uint32_t, CarState>> seq;
		for (int r = 0; r < 12; r++) {
			setter.ResetState(arena);
			std::map<uint32_t, CarState> byId;
			for (Car* c : arena->_cars) byId[c->id] = c->GetState();
			seq.push_back(byId);
		}
		return seq;
	};
	auto ref = carStates(arenas[0]);
	for (size_t k = 1; k < arenas.size(); k++) {
		auto seq = carStates(arenas[k]);
		for (size_t r = 0; r < seq.size(); r++) {
			for (auto& [id, cs] : ref[r]) {
				const CarState& other = seq[r].at(id);
				if ((cs.pos - other.pos).Length() > 1e-3f || std::abs(cs.boost - other.boost) > 1e-3f)
					FAIL_AT("Arena " << k << ", Reset " << r << ", Car-ID " << id
					        << ": anderer Zustand bei gleichem Seed (Reihenfolge von arena->_cars)");
			}
		}
	}
	for (Arena* a : arenas) delete a;
}
