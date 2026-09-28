// Spielanalyse (env/cpp/PlayStats.h) und ihre Trainingsmetriken (train/cpp/Metrics.cpp), geprüft am
// echten Pfad: Arena -> Gym::Step -> PlayTracker -> AccumPlayMetrics -> AggregateGameMetrics.
// Die Szenen werden mit einem State-Setter bzw. SetState in der Arena gebaut, nicht als GameState von Hand.
#include "test_util.h"

#include "env/cpp/Obs.h"
#include "env/cpp/PlayStats.h"
#include "env/cpp/StateSetters.h"
#include "train/cpp/Metrics.h"

#include <RLGymSim_CPP/Gym.h>
#include <RLGymSim_CPP/Math.h>
#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>

#include <functional>

using namespace RLbot;
using RLGPC::Report;

extern bool g_arenaReady;

namespace {

class FuncSetter : public StateSetter {
public:
	std::function<void(Arena*)> fn;
	explicit FuncSetter(std::function<void(Arena*)> fn) : fn(std::move(fn)) {}
	virtual GameState ResetState(Arena* arena) {
		fn(arena);
		return GameState(arena);
	}
};

// Index des ersten Tabelleneintrags mit diesen Werten (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)
int ActionIndex(std::array<float, 8> want) {
	DiscreteAction parser;
	for (size_t i = 0; i < parser.actions.size(); i++) {
		bool same = true;
		for (int e = 0; e < 8; e++)
			same &= parser.actions[i][e] == want[e];
		if (same)
			return (int)i;
	}
	throw std::runtime_error("Aktion nicht in der Tabelle");
}
const int DRIVE_BOOST = ActionIndex({ 1, 0, 0, 0, 0, 0, 1, 0 });
const int IDLE = ActionIndex({ 0, 0, 0, 0, 0, 0, 0, 0 });

struct Env {
	Match* match;
	Gym* gym;
	Env(StateSetter* setter, int teamSize) {
		match = new Match(new EventReward({}), { new GoalScoreCondition() }, new StackedPaddedOBS(3, 5, false),
		                  new DiscreteAction(), setter, teamSize, true);
		gym = new Gym(match, 8);
	}
	~Env() { delete gym; delete match; }
	// Aktionen je Car-ID (Reihenfolge von state.players folgt arena->_cars, adressabhängig, R19)
	Gym::StepResult Step(std::function<int(const PlayerData&)> act) {
		const auto& players = gym->prevState.players;
		IList actions(players.size());
		for (size_t i = 0; i < players.size(); i++)
			actions[i] = act(players[i]);
		return gym->Step(actions);
	}
};

void Append(PlayEvents& all, const PlayEvents& ev) {
	all.kickoffs.insert(all.kickoffs.end(), ev.kickoffs.begin(), ev.kickoffs.end());
	all.spells.insert(all.spells.end(), ev.spells.begin(), ev.spells.end());
	all.touches.insert(all.touches.end(), ev.touches.begin(), ev.touches.end());
	all.teams.insert(all.teams.end(), ev.teams.begin(), ev.teams.end());
	all.shots[0] += ev.shots[0];
	all.shots[1] += ev.shots[1];
	if (ev.goalTeam >= 0)
		all.goalTeam = ev.goalTeam;
}

} // namespace

TEST(Spielanalyse_Anstoss_Blau_faehrt_mit_Boost_Orange_steht) {
	if (!g_arenaReady) return;
	// Anstoß hinten Mitte (RocketSim-Position 0, -4608): Blau fährt geradeaus mit Boost auf den Ball,
	// Orange steht. Von der Diagonalen aus würde Geradeausfahren den Ball verfehlen (Blick 45 Grad, Ball 51).
	Env env(new FuncSetter([](Arena* a) {
		a->ResetToRandomKickoff(0);
		for (Car* car : a->_cars) {
			CarState cs = {};
			bool blue = car->team == Team::BLUE;
			cs.pos = Vec(0, blue ? -4608.f : 4608.f, 17);
			cs.rotMat = Angle(blue ? M_PI / 2 : -M_PI / 2, 0, 0).ToRotMat();
			cs.boost = 100 / 3.f;
			car->SetState(cs);
		}
	}), 1);
	env.gym->Reset();
	PlayTracker tracker;
	PlayEvents all;
	Report metrics;
	for (int i = 0; i < 160 && all.kickoffs.empty(); i++) {
		auto r = env.Step([](const PlayerData& p) { return p.team == Team::BLUE ? DRIVE_BOOST : IDLE; });
		auto ev = tracker.Step(r.state, r.done);
		AccumPlayMetrics(ev, 1, 8, metrics);
		Append(all, ev);
		if (r.done) break;
	}
	CHECK_EQ(all.kickoffs.size(), (size_t)1);
	const auto& ko = all.kickoffs[0];
	CHECK_EQ(ko.firstTeam, 0);
	CHECK_GT(ko.timeToTouch, 1.5);
	CHECK_GT(3.2, ko.timeToTouch);
	CHECK_GT(ko.touchSpeed, 1500);
	CHECK_NEAR(ko.loserSpeed, 0, 5);
	CHECK_GT(ko.boostUsed[0], 20);                  // 33 Boost beim Anstoß, fast alles verbraucht
	CHECK_NEAR(ko.boostUsed[1], 0, 1e-6);
	CHECK_EQ(ko.ballHalfTeam, 0);                   // Ball ist in Oranges Hälfte
	// Erste Berührung tickgenau: stimmt mit dem ersten Kontakt-Schritt überein (Fenster 8 Ticks)
	CHECK(!all.touches.empty());
	CHECK_EQ(all.touches[0].team, 0);
	CHECK(!all.touches[0].carInAir);

	Report agg;
	AggregateGameMetrics({ metrics }, agg);
	CHECK_NEAR(agg["kickoff_first_touch_s"], ko.timeToTouch, 1e-5);
	CHECK_NEAR(agg["kickoff_touch_speed"], ko.touchSpeed, 1e-3);
	CHECK_NEAR(agg["kickoff_boost_used"], ko.boostUsed[0] / 2.0, 1e-4);
	CHECK_NEAR(agg["kickoff_untouched"], 0, 1e-9);
	CHECK(!agg.Has("2v2_kickoff_first_touch_s"));
}

TEST(Spielanalyse_kein_Anstoss_wenn_der_Ball_nicht_ruht) {
	if (!g_arenaReady) return;
	StateSetterWeights w = {};
	w.kickoff = 0; w.random = 0; w.aerial = 1;
	Env env(new WeightedStateSetter(w, 5), 1);
	env.gym->Reset();
	PlayTracker tracker;
	PlayEvents all;
	for (int i = 0; i < 200; i++)
		Append(all, tracker.Step(env.Step([](const PlayerData&) { return IDLE; }).state, false));
	Append(all, tracker.Flush());
	CHECK(all.kickoffs.empty());
}

TEST(Spielanalyse_Luftberuehrung_mit_Hoehe_und_Bodenkontakt) {
	if (!g_arenaReady) return;
	Env env(new FuncSetter([](Arena* a) {
		a->ResetToRandomKickoff(0);
		BallState bs = {};
		bs.pos = Vec(0, 0, 700);
		bs.vel = Vec(0, 0, -200);
		a->ball->SetState(bs);
		for (Car* car : a->_cars) {
			CarState cs = {};
			cs.rotMat = RotMat::GetIdentity();
			if (car->team == Team::BLUE) {
				cs.pos = Vec(0, 0, 520);                     // unter dem Ball, in der Luft, steigt
				cs.vel = Vec(0, 0, 500);
				cs.isOnGround = false;
				cs.hasJumped = true;
			} else {
				cs.pos = Vec(3000, 3000, 17);
			}
			car->SetState(cs);
		}
	}), 1);
	env.gym->Reset();
	PlayTracker tracker;
	PlayEvents all;
	Report metrics;
	for (int i = 0; i < 10; i++) {
		auto ev = tracker.Step(env.Step([](const PlayerData&) { return IDLE; }).state, false);
		AccumPlayMetrics(ev, 1, 8, metrics);
		Append(all, ev);
	}
	CHECK_EQ(all.touches.size(), (size_t)1);
	CHECK(all.touches[0].carInAir);
	CHECK_GT(all.touches[0].ballHeight, AERIAL_TOUCH_MIN_HEIGHT);
	Report agg;
	AggregateGameMetrics({ metrics }, agg);
	// 1 Luftkontakt in 10 Steps bei 2 Spielern: je Spieler 0,5 Kontakte in 10/900 min = 45 pro Minute
	CHECK_NEAR(agg["air_touch_per_min"], 45.0, 1e-6);
	CHECK_NEAR(agg["aerial_touch_per_min"], agg["air_touch_per_min"], 1e-9);
	CHECK_NEAR(agg["air_touch_share"], 1, 1e-9);
	CHECK_NEAR(agg["touch_height_mean"], all.touches[0].ballHeight, 1e-3);
}

TEST(Spielanalyse_Angriffsdrittel_Aufenthalt_ohne_und_mit_Tor) {
	if (!g_arenaReady) return;
	// Ball ruht in Blaus Angriffsdrittel, beide Autos weit weg und still
	Env env(new FuncSetter([](Arena* a) {
		a->ResetToRandomKickoff(0);
		BallState bs = {};
		bs.pos = Vec(2500, 3500, 93);
		a->ball->SetState(bs);
		for (Car* car : a->_cars) {
			CarState cs = {};
			cs.rotMat = RotMat::GetIdentity();
			cs.pos = Vec(car->team == Team::BLUE ? -3000.f : 3000.f, -3000, 17);
			car->SetState(cs);
		}
	}), 1);
	env.gym->Reset();
	PlayTracker tracker;
	PlayEvents all;
	Report metrics;
	auto step = [&](int n) {
		for (int i = 0; i < n; i++) {
			auto r = env.Step([](const PlayerData&) { return IDLE; });
			auto ev = tracker.Step(r.state, r.done);
			AccumPlayMetrics(ev, 1, 8, metrics);
			Append(all, ev);
		}
	};
	step(30);                                          // 2 s im Drittel
	BallState mid = {};
	mid.pos = Vec(0, 0, 93);
	env.gym->arena->ball->SetState(mid);
	step(10);                                          // < 1 s draußen: Aufenthalt läuft noch
	CHECK(all.spells.empty());
	step(10);                                          // jetzt > 1 s draußen: abgeschlossen
	CHECK_EQ(all.spells.size(), (size_t)1);
	CHECK_EQ(all.spells[0].team, 0);
	CHECK(!all.spells[0].goal);
	CHECK_NEAR(all.spells[0].seconds, 30 * 8 / 120.0, 1e-4);

	// Schuss: Ball rollt schnell auf das orange Tor, Blau hat ihn gerade berührt
	BallState shot = {};
	shot.pos = Vec(0, 3000, 93);
	env.gym->arena->ball->SetState(shot);
	for (Car* car : env.gym->arena->_cars) {
		if (car->team != Team::BLUE) continue;
		CarState cs = {};
		cs.rotMat = Angle(M_PI / 2, 0, 0).ToRotMat();
		cs.pos = Vec(0, 2780, 17);
		cs.vel = Vec(0, 2200, 0);
		car->SetState(cs);
	}
	Gym::StepResult r = {};
	for (int i = 0; i < 90 && !r.done; i++) {
		r = env.Step([](const PlayerData& p) { return p.team == Team::BLUE ? DRIVE_BOOST : IDLE; });
		auto ev = tracker.Step(r.state, r.done);
		AccumPlayMetrics(ev, 1, 8, metrics);
		Append(all, ev);
	}
	CHECK(r.done);
	CHECK_EQ(all.goalTeam, 0);
	CHECK_EQ(all.shots[0], 1);
	CHECK_EQ(all.shots[1], 0);
	CHECK_EQ(all.spells.size(), (size_t)2);
	CHECK(all.spells[1].goal);

	Report agg;
	AggregateGameMetrics({ metrics }, agg);
	CHECK_NEAR(agg["off_third_conversion"], 0.5, 1e-9);
	CHECK_NEAR(agg["off_third_nogoal_s"], all.spells[0].seconds, 1e-4);
	CHECK_NEAR(agg["off_third_long"], 0, 1e-9);
	CHECK_GT(agg["goals_per_min"], 0);
	CHECK_GT(agg["shots_per_min"], 0);
}

TEST(Spielanalyse_Team_2v2_Double_Commit_Absicherung_und_Praefix) {
	if (!g_arenaReady) return;
	// Blau: beide Autos knapp vor dem eigenen Tor-Rand des Balls (y = -300, weniger als BACK_MARGIN dahinter)
	// und fahren auf ihn zu; Orange: eins hinter dem Ball (y = 3000, sichert ab), eins weit davor
	Env env(new FuncSetter([](Arena* a) {
		a->ResetToRandomKickoff(0);
		BallState bs = {};
		bs.pos = Vec(0, 0, 93);
		a->ball->SetState(bs);
		std::vector<Car*> cars(a->_cars.begin(), a->_cars.end());
		std::sort(cars.begin(), cars.end(), [](Car* x, Car* y) { return x->id < y->id; });
		int blue = 0, orange = 0;
		for (Car* car : cars) {
			CarState cs = {};
			if (car->team == Team::BLUE) {
				float x = blue++ ? 600.f : -600.f;
				cs.pos = Vec(x, -300, 17);
				Vec dir = (Vec(0, 0, 17) - cs.pos).Normalized();
				cs.vel = dir * 1200;
				cs.rotMat = Angle(std::atan2(dir.y, dir.x), 0, 0).ToRotMat();
			} else {
				cs.pos = orange++ ? Vec(0, 3000, 17) : Vec(0, -3000, 17);   // zweites Orange-Auto vor dem Ball
				cs.rotMat = Angle(-M_PI / 2, 0, 0).ToRotMat();
			}
			car->SetState(cs);
		}
	}), 2);
	env.gym->Reset();
	PlayTracker tracker;
	Report metrics;
	auto r = env.Step([](const PlayerData&) { return IDLE; });
	auto ev = tracker.Step(r.state, r.done);
	AccumPlayMetrics(ev, 2, 8, metrics);
	CHECK_EQ(ev.teams.size(), (size_t)2);
	for (auto& t : ev.teams) {
		if (t.team == 0) {
			CHECK(t.doubleCommit);
			CHECK(!t.lastBack);                           // keiner mehr als BACK_MARGIN hinter dem Ball
			CHECK_NEAR(t.mateDist, 1200, 60);
		} else {
			CHECK(!t.doubleCommit);
			CHECK(t.lastBack);                            // Orange verteidigt +y: Auto bei y = 3000
			CHECK_NEAR(t.mateDist, 6000, 60);
		}
	}
	Report agg;
	AggregateGameMetrics({ metrics }, agg);
	CHECK(agg.Has("2v2_double_commit"));
	CHECK_NEAR(agg["2v2_double_commit"], 0.5, 1e-9);
	CHECK(agg.Has("2v2_air_touch_per_min"));
	CHECK(!agg.Has("double_commit"));
	CHECK(!agg.Has("air_touch_per_min"));
}

TEST(Spielanalyse_Tracker_je_Spiel_getrennt) {
	int a = 0, b = 0;
	PlayTracker& ta = PlayTrackerFor(&a, 8);
	CHECK(&PlayTrackerFor(&a, 8) == &ta);
	CHECK(&PlayTrackerFor(&b, 8) != &ta);
}
