// Neue Rewards und die Anstoß-Drill-Szene für die Spieltest-Experimente, geprüft am echten Pfad
// (Arena -> Gym::Step -> Match-Reward bzw. EnvFactory -> Terminal-Bedingungen -> Metrik).
#include "test_util.h"

#include "env/cpp/Obs.h"
#include "env/cpp/Rewards.h"
#include "env/cpp/StateSetters.h"
#include "env/cpp/TimeoutCondition.h"
#include "train/cpp/Config.h"
#include "train/cpp/EnvFactory.h"
#include "train/cpp/Metrics.h"

#include <RLGymSim_CPP/Gym.h>
#include <RLGymSim_CPP/Math.h>
#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>

#include <filesystem>
#include <fstream>
#include <functional>

using namespace RLbot;

extern bool g_arenaReady;

namespace {

class FuncSetter : public StateSetter {
public:
	std::function<void(Arena*)> fn;
	explicit FuncSetter(std::function<void(Arena*)> fn) : fn(std::move(fn)) {}
	virtual GameState ResetState(Arena* arena) { fn(arena); return GameState(arena); }
};

int ActionIndex(std::array<float, 8> want) {
	DiscreteAction parser;
	for (size_t i = 0; i < parser.actions.size(); i++) {
		bool same = true;
		for (int e = 0; e < 8; e++) same &= parser.actions[i][e] == want[e];
		if (same) return (int)i;
	}
	throw std::runtime_error("Aktion nicht in der Tabelle");
}
const int DRIVE_BOOST = ActionIndex({ 1, 0, 0, 0, 0, 0, 1, 0 });
const int IDLE = ActionIndex({ 0, 0, 0, 0, 0, 0, 0, 0 });

RewardWeights Only(std::function<void(RewardWeights&)> set) {
	RewardWeights w = {};
	w.goal = 0; w.concede = 0; w.touchBallToGoalAccel = 0; w.offensivePotential = 0; w.distWeightedAlign = 0;
	w.velocityPlayerToBall = 0; w.saveBoost = 0; w.inAir = 0; w.teamSpirit = 0;
	set(w);
	return w;
}

// Anstoß hinten Mitte, Blau fährt mit Boost, Orange steht (wie test_play_stats.cpp)
void BackCenterKickoff(Arena* a) {
	a->ResetToRandomKickoff(0);
	for (Car* car : a->_cars) {
		CarState cs = {};
		bool blue = car->team == Team::BLUE;
		cs.pos = Vec(0, blue ? -4608.f : 4608.f, 17);
		cs.rotMat = Angle(blue ? M_PI / 2 : -M_PI / 2, 0, 0).ToRotMat();
		cs.boost = 100 / 3.f;
		car->SetState(cs);
	}
}

struct Run {
	Match* match;
	Gym* gym;
	Run(RewardFunction* reward, StateSetter* setter, std::vector<TerminalCondition*> conds = { new GoalScoreCondition() }) {
		match = new Match(reward, conds, new StackedPaddedOBS(3, 5, false), new DiscreteAction(), setter, 1, true);
		gym = new Gym(match, 8);
		gym->Reset();
	}
	~Run() { delete gym; delete match; }
	// Rewards je Team (0 = Blau) in diesem Schritt
	std::array<float, 2> Step(int blueAction, int orangeAction, Gym::StepResult* out = nullptr) {
		const auto& players = gym->prevState.players;
		IList actions(players.size());
		for (size_t i = 0; i < players.size(); i++)
			actions[i] = players[i].team == Team::BLUE ? blueAction : orangeAction;
		auto r = gym->Step(actions);
		std::array<float, 2> byTeam = { 0, 0 };
		for (size_t i = 0; i < r.state.players.size(); i++)
			byTeam[r.state.players[i].team == Team::BLUE ? 0 : 1] = r.reward[i];
		if (out) *out = r;
		return byTeam;
	}
};

} // namespace

TEST(Spieltest_Anstoss_erste_Beruehrung_zahlt_einmal_dem_Schnelleren) {
	if (!g_arenaReady) return;
	for (float tau : { 0.f, 0.1f }) {
		Run run(BuildLucyReward(Only([tau](RewardWeights& w) { w.kickoffFirstTouch = 2; w.teamSpirit = tau; })),
		        new FuncSetter(BackCenterKickoff));
		float blueSum = 0, orangeSum = 0;
		int paidSteps = 0;
		for (int i = 0; i < 100; i++) {
			auto r = run.Step(DRIVE_BOOST, IDLE);
			blueSum += r[0];
			orangeSum += r[1];
			paidSteps += r[0] != 0;
		}
		CHECK_EQ(paidSteps, 1);                       // Blau war am Ball, danach nie wieder
		CHECK_NEAR(blueSum, 2.0, 1e-6);
		CHECK_NEAR(orangeSum, tau > 0 ? -2.0 : 0.0, 1e-6);   // Zero-Sum: der Gegner verliert dasselbe
	}
}

TEST(Spieltest_Anstoss_Reward_nur_nach_einem_Anstoss) {
	if (!g_arenaReady) return;
	// Ball liegt nicht in der Mitte: kein Anstoß, auch eine Berührung zahlt nichts
	Run run(BuildLucyReward(Only([](RewardWeights& w) { w.kickoffFirstTouch = 2; })), new FuncSetter([](Arena* a) {
		BackCenterKickoff(a);
		BallState bs = {};
		bs.pos = Vec(0, -3500, 93);
		a->ball->SetState(bs);
	}));
	float sum = 0;
	for (int i = 0; i < 60; i++) {
		auto r = run.Step(DRIVE_BOOST, IDLE);
		sum += std::abs(r[0]) + std::abs(r[1]);
	}
	CHECK_NEAR(sum, 0.0, 1e-9);
}

TEST(Spieltest_Potenzial_Halten_kostet_Verbessern_zahlt_Summe_teleskopiert) {
	if (!g_arenaReady) return;
	const float gamma = 0.9954f;
	auto setup = [](Arena* a) {
		a->ResetToRandomKickoff(0);
		BallState bs = {};
		bs.pos = Vec(0, 3000, 93);
		a->ball->SetState(bs);
		for (Car* car : a->_cars) {
			CarState cs = {};
			bool blue = car->team == Team::BLUE;
			cs.pos = blue ? Vec(0, 1000, 17) : Vec(3000, 4000, 17);
			cs.rotMat = Angle(M_PI / 2, 0, 0).ToRotMat();
			car->SetState(cs);
		}
	};
	// Direkte Form: Stillstand hinter dem Ball zahlt jeden Schritt
	Run direct(BuildLucyReward(Only([](RewardWeights& w) { w.distWeightedAlign = 1; })), new FuncSetter(setup));
	auto d = direct.Step(IDLE, IDLE);
	CHECK_GT(d[0], 0.3);
	// Potenzialform: Stillstand kostet (gamma - 1) * Phi, also leicht negativ
	Run pot(BuildLucyReward(Only([gamma](RewardWeights& w) {
		w.distWeightedAlign = 1; w.potentialShapingScale = 1; w.potentialGamma = gamma; })), new FuncSetter(setup));
	auto p = pot.Step(IDLE, IDLE);
	CHECK_NEAR(p[0], (gamma - 1) * d[0], 1e-4);
	// Auf den Ball zufahren: diskontierte Summe = gamma^T * Phi(s_T) - Phi(s_0)
	double disc = 0, g = 1;
	int steps = 12;                                         // 0,8 s: Anfahrt, noch kein Kontakt
	for (int i = 0; i < steps; i++) {
		auto r = pot.Step(DRIVE_BOOST, IDLE);
		disc += g * r[0];
		g *= gamma;
	}
	CHECK_GT(disc, 0);                                      // Lage verbessert: positiv
	RewardFunction* phi = MakeDistWeightedAlignment();
	const auto& s = pot.gym->prevState;
	float phiEnd = 0;
	for (auto& pl : s.players) if (pl.team == Team::BLUE) phiEnd = phi->GetReward(pl, s, Action());
	// Phi(s_0) der Teilsumme ist Phi nach dem ersten (Halte-)Schritt = d[0]
	CHECK_NEAR(disc, g * phiEnd - d[0], 1e-3);
	delete phi;
}

TEST(Spieltest_Potenzial_am_Tor_faellt_auf_null) {
	if (!g_arenaReady) return;
	// Ball fliegt ins orange Tor, Blau steht hinter dem Ball (hohes Potenzial): Torschritt = -Phi(vorher)
	Run run(BuildLucyReward(Only([](RewardWeights& w) {
		w.distWeightedAlign = 1; w.potentialShapingScale = 1; w.potentialGamma = 0.9954f; })),
		new FuncSetter([](Arena* a) {
			a->ResetToRandomKickoff(0);
			BallState bs = {};
			bs.pos = Vec(0, 4600, 93);
			bs.vel = Vec(0, 2500, 0);
			a->ball->SetState(bs);
			for (Car* car : a->_cars) {
				CarState cs = {};
				cs.pos = car->team == Team::BLUE ? Vec(0, 3800, 17) : Vec(3000, -4000, 17);
				cs.rotMat = Angle(M_PI / 2, 0, 0).ToRotMat();
				car->SetState(cs);
			}
		}));
	RewardFunction* phi = MakeDistWeightedAlignment();
	Gym::StepResult r = {};
	float lastPhi = 0, lastReward = 0;
	for (int i = 0; i < 30 && !r.done; i++) {
		for (auto& pl : run.gym->prevState.players)
			if (pl.team == Team::BLUE) lastPhi = phi->GetReward(pl, run.gym->prevState, Action());
		lastReward = run.Step(IDLE, IDLE, &r)[0];
	}
	CHECK(r.done);
	CHECK(RLGSC::Math::IsBallScored(r.state.ball.pos));
	CHECK_NEAR(lastReward, -lastPhi, 1e-4);
	delete phi;
}

TEST(Spieltest_Luftberuehrung_mit_Hoehe_und_Sperrzeit) {
	if (!g_arenaReady) return;
	auto setup = [](Arena* a) {
		a->ResetToRandomKickoff(0);
		BallState bs = {};
		bs.pos = Vec(0, 0, 700);
		bs.vel = Vec(0, 0, -200);
		a->ball->SetState(bs);
		for (Car* car : a->_cars) {
			CarState cs = {};
			cs.rotMat = RotMat::GetIdentity();
			if (car->team == Team::BLUE) {
				cs.pos = Vec(0, 0, 520);
				cs.vel = Vec(0, 0, 500);
				cs.isOnGround = false;
				cs.hasJumped = true;
			} else {
				cs.pos = Vec(3000, 3000, 17);
			}
			car->SetState(cs);
		}
	};
	Run run(BuildLucyReward(Only([](RewardWeights& w) { w.airTouch = 1; })), new FuncSetter(setup));
	float sum = 0, paid = 0;
	uint64_t hitTick = 0;
	float hitZ = 0;
	for (int i = 0; i < 10; i++) {
		Gym::StepResult r = {};
		auto rw = run.Step(IDLE, IDLE, &r);
		if (rw[0] > 0) {
			paid++;
			for (auto& pl : r.state.players)
				if (pl.team == Team::BLUE) { hitTick = pl.carState.ballHitInfo.tickCountWhenHit; hitZ = pl.carState.ballHitInfo.ballPos.z; }
		}
		sum += rw[0];
	}
	CHECK_EQ(paid, 1.0f);
	CHECK_NEAR(sum, (hitZ - CommonValues::BALL_RADIUS) / (CommonValues::CEILING_Z - CommonValues::BALL_RADIUS), 1e-5);
	CHECK_GT(sum, 0.2);
	CHECK(hitTick > 0);

	// Sperrzeit: direkt der AirTouchReward auf zwei Kontakte desselben Autos 30 Ticks auseinander
	AirTouchReward air;
	GameState s = run.gym->prevState;
	PlayerData* blue = nullptr;
	for (auto& pl : s.players) if (pl.team == Team::BLUE) blue = &pl;
	air.Reset(s);
	blue->ballTouchedStep = true;
	blue->carState.isOnGround = false;
	blue->carState.ballHitInfo.isValid = true;
	blue->carState.ballHitInfo.ballPos = Vec(0, 0, 1000);
	blue->carState.ballHitInfo.tickCountWhenHit = 1000;
	CHECK_GT(air.GetReward(*blue, s, Action()), 0);
	blue->carState.ballHitInfo.tickCountWhenHit = 1030;
	CHECK_NEAR(air.GetReward(*blue, s, Action()), 0, 1e-9);        // innerhalb 0,5 s
	blue->carState.ballHitInfo.tickCountWhenHit = 1000 + AirTouchReward::COOLDOWN_TICKS;
	CHECK_GT(air.GetReward(*blue, s, Action()), 0);
	blue->carState.isOnGround = true;
	blue->carState.ballHitInfo.tickCountWhenHit = 2000;
	CHECK_NEAR(air.GetReward(*blue, s, Action()), 0, 1e-9);        // am Boden nie
}

TEST(Spieltest_Anstoss_Drill_wird_nach_der_eingestellten_Zeit_abgeschnitten) {
	if (!g_arenaReady) return;
	TrainConfig cfg = {};
	cfg.states = {};
	cfg.states.kickoff = 0; cfg.states.random = 0; cfg.states.kickoffDrill = 1;
	cfg.kickoffDrillSecs = 2;                     // 30 Schritte
	cfg.gameTimeoutSecs = 1000; cfg.noTouchTimeoutSecs = 1000;
	cfg.seedEnvs = true;
	EnvFactory factory(cfg);
	auto env = factory.Create();
	env.gym->Reset();
	CHECK_EQ(CurrentSceneName(env.match), std::string("kickoff_drill"));
	Gym::StepResult r = {};
	int steps = 0;
	while (!r.done && steps < 100) {
		r = env.gym->Step(IList(env.gym->prevState.players.size(), IDLE));
		steps++;
	}
	CHECK_EQ(steps, 30);
	CHECK(r.truncated);                            // Welt läuft weiter: Bootstrap (K1b)
	auto end = ClassifyEpisodeEnd(r.state, env.match, r.truncated);
	CHECK(end.drill && !end.goal && !end.timeLimit && !end.noTouch);
	RLGPC::Report m;
	AccumEpisodeEnd(end, steps, "kickoff_drill", m);
	CHECK_NEAR(m.GetAvg("ep_end_drill"), 1, 1e-9);
	CHECK_NEAR(m.GetAvg("ep_end_timeout"), 0, 1e-9);
	CHECK_NEAR(m.GetAvg("scene_kickoff_drill_length"), 30, 1e-9);
	delete env.gym; delete env.match;

	// Ohne Drill-Szene keine zusätzliche Bedingung, andere Szenen werden nie abgeschnitten
	cfg.states.kickoffDrill = 0; cfg.states.kickoff = 1;
	EnvFactory plain(cfg);
	auto e2 = plain.Create();
	CHECK_EQ(e2.match->terminalConditions.size(), (size_t)3);
	delete e2.gym; delete e2.match;
}

TEST(Spieltest_Config_liest_neue_Felder_und_Standard_ist_aus) {
	TrainConfig def = {};
	CHECK_EQ(def.rewards.kickoffFirstTouch, 0.f);
	CHECK_EQ(def.rewards.airTouch, 0.f);
	CHECK_EQ(def.rewards.potentialShapingScale, 0.f);
	CHECK_EQ(def.states.kickoffDrill, 0.f);
	// Standardgewichte (wie lucy_1v1): dieselben 7 Summanden wie vorher, direkte Form
	auto parts = BuildLucyRewardParts(RewardWeights{});
	std::vector<std::string> names;
	for (auto& p : parts) { names.push_back(p.name); delete p.fn; }
	CHECK_EQ(names.size(), (size_t)7);
	CHECK_EQ(names[2], std::string("offensive_potential_krc"));
	CHECK_EQ(names[6], std::string("in_air"));

	auto path = std::filesystem::temp_directory_path() / "rlbot_test_spieltest_cfg.json";
	std::ofstream(path) << R"({
		"env": { "kickoff_drill_secs": 5.0 },
		"rewards": { "kickoff_first_touch": 2.0, "air_touch": 1.5, "potential_shaping_scale": 10.0 },
		"state_setters": { "kickoff_drill": 4.0 },
		"learner": { "gae_gamma": 0.99 }
	})";
	auto cfg = TrainConfig::FromFile(path.string());
	CHECK_EQ(cfg.kickoffDrillSecs, 5.f);
	CHECK_EQ(cfg.rewards.kickoffFirstTouch, 2.f);
	CHECK_EQ(cfg.rewards.airTouch, 1.5f);
	CHECK_EQ(cfg.rewards.potentialShapingScale, 10.f);
	CHECK_NEAR(cfg.rewards.potentialGamma, 0.99, 1e-7);    // gleiches gamma wie die GAE
	CHECK_EQ(cfg.states.kickoffDrill, 4.f);
	std::ofstream(path) << cfg.ToJSONString();             // config_used.json ist wieder ladbar
	auto again = TrainConfig::FromFile(path.string());
	CHECK_EQ(again.rewards.potentialShapingScale, 10.f);
	CHECK_EQ(again.states.kickoffDrill, 4.f);
	CHECK_EQ(again.kickoffDrillSecs, 5.f);

	auto potParts = BuildLucyRewardParts(cfg.rewards);
	bool sawPot = false;
	for (auto& p : potParts) {
		if (p.name == "offensive_potential_krc_pot") {
			sawPot = true;
			CHECK_NEAR(p.weight, 10.0, 1e-6);                   // Gewicht 1 mal Faktor 10
			CHECK(dynamic_cast<PotentialReward*>(p.fn) != nullptr);
		}
		delete p.fn;
	}
	CHECK(sawPot);

	std::ofstream(path) << R"({"rewards": {"potential_shaping_scale": -1}})";
	bool failed = false;
	try { TrainConfig::FromFile(path.string()); } catch (const std::exception&) { failed = true; }
	CHECK(failed);
	std::filesystem::remove(path);
}
