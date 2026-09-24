// Prüft die Lucy-nahen Rewards gegen von Hand nachgerechnete Werte.
#include "test_util.h"

#include "env/cpp/Rewards.h"

#include <RLGymSim_CPP/Utils/RewardFunctions/CombinedReward.h>

using namespace RLGSC;
using namespace RLbot;

// --- Hilfsmittel ---------------------------------------------------------

static PlayerData MakePlayer(uint32_t id, Team team, Vec pos, Vec vel = {}, float boost = 1.f) {
	PlayerData p = {};
	p.carId = id;
	p.team = team;
	p.phys.pos = pos;
	p.phys.vel = vel;
	p.phys.rotMat = RotMat::GetIdentity();
	p.physInv = p.phys.Invert();
	p.carState.pos = pos;
	p.carState.vel = vel;
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

// Reward-Funktion mit festem Wert, um die KRC isoliert zu prüfen.
class ConstReward : public RewardFunction {
public:
	float val;
	explicit ConstReward(float val) : val(val) {}
	virtual float GetReward(const PlayerData&, const GameState&, const Action&) { return val; }
};

static float Single(RewardFunction* fn, const GameState& state, int playerIdx = 0) {
	Action empty = {};
	fn->Reset(state);
	fn->PreStep(state);
	return fn->GetAllRewards(state, ActionSet(state.players.size()), false)[playerIdx];
}

// --- KRC -----------------------------------------------------------------

TEST(KRC_geometrisches_Mittel) {
	auto state = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, 17)) }, Vec(0, 500, 93));
	Action empty = {};

	KRCReward krc({ new ConstReward(0.25f), new ConstReward(1.0f) });
	// sqrt(0.25 * 1.0) = 0.5
	CHECK_NEAR(krc.GetReward(state.players[0], state, empty), 0.5, 1e-6);

	KRCReward krc3({ new ConstReward(0.5f), new ConstReward(0.5f), new ConstReward(0.5f) });
	CHECK_NEAR(krc3.GetReward(state.players[0], state, empty), 0.5, 1e-6);
}

TEST(KRC_Vorzeichen_negativ_wenn_eine_Komponente_negativ) {
	auto state = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, 17)) }, Vec(0, 500, 93));
	Action empty = {};

	KRCReward krc({ new ConstReward(-0.5f), new ConstReward(0.5f) });
	CHECK_NEAR(krc.GetReward(state.players[0], state, empty), -0.5, 1e-6);

	// Beide negativ: Betrag identisch, Vorzeichen bleibt negativ (nur "alle positiv" ist positiv)
	KRCReward bothNeg({ new ConstReward(-0.5f), new ConstReward(-0.5f) });
	CHECK_NEAR(bothNeg.GetReward(state.players[0], state, empty), -0.5, 1e-6);
}

TEST(KRC_eine_Null_Komponente_macht_alles_Null) {
	auto state = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, 17)) }, Vec(0, 500, 93));
	Action empty = {};
	KRCReward krc({ new ConstReward(0.f), new ConstReward(1.f) });
	CHECK_NEAR(krc.GetReward(state.players[0], state, empty), 0.0, 1e-9);
}

// --- Distanz -------------------------------------------------------------

TEST(Distanz_Reward_Formel_und_Monotonie) {
	// exp(-0.5 * d / (scale * dispersion)) ^ (1 / density)
	CHECK_NEAR(ParamDistanceReward::Compute(0.f, 2300.f, 1.f, 1.f), 1.0, 1e-9);
	CHECK_NEAR(ParamDistanceReward::Compute(2300.f, 2300.f, 1.f, 1.f), std::exp(-0.5), 1e-6);
	CHECK_NEAR(ParamDistanceReward::Compute(2300.f, 2300.f, 1.f, 2.f), std::exp(-0.25), 1e-6);
	CHECK_NEAR(ParamDistanceReward::Compute(2300.f, 2300.f, 2.f, 1.f), std::exp(-0.25), 1e-6);

	float prev = 2.f;
	for (float d = 0; d <= 8000; d += 250) {
		float r = ParamDistanceReward::Compute(d, 2300.f, 1.f, 1.f);
		CHECK(r < prev);
		CHECK(r > 0);
		prev = r;
	}
}

TEST(Distanz_Spieler_Ball_zieht_Ballradius_ab) {
	// Auto direkt am Ball: Abstand der Mittelpunkte = BALL_RADIUS -> Distanz 0 -> Reward 1
	auto state = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, 93)) },
	                       Vec(0, CommonValues::BALL_RADIUS, 93));
	ParamDistanceReward r(ParamDistanceReward::Target::PLAYER_TO_BALL, CommonValues::CAR_MAX_SPEED);
	CHECK_NEAR(Single(&r, state), 1.0, 1e-5);
}

// --- Ausrichtung ---------------------------------------------------------

// Die Tormittelpunkte liegen auf GOAL_HEIGHT/2. Auto und Ball auf dieselbe Höhe zu setzen
// macht die Richtungsvektoren exakt achsenparallel, damit sind die Erwartungswerte exakt 1
// statt 0,994 (der Rest wäre nur die z-Komponente des Richtungsvektors).
static constexpr float GOAL_Z = CommonValues::GOAL_HEIGHT / 2;

TEST(AlignBallGoal_perfekte_Ausrichtung) {
	// Blau spielt auf +Y. Auto bei y=0, Ball bei y=1000 -> Vektor Spieler->Ball zeigt aufs Tor
	auto state = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, GOAL_Z)) }, Vec(0, 1000, GOAL_Z));
	AlignBallGoalReward r(1.f, 0.f);
	CHECK_NEAR(Single(&r, state), 1.0, 1e-5);

	// Gleiche Geometrie für Orange: Orange spielt auf -Y, also falsch herum
	auto state2 = MakeState({ MakePlayer(1, Team::ORANGE, Vec(0, 0, GOAL_Z)) }, Vec(0, 1000, GOAL_Z));
	CHECK_NEAR(Single(&r, state2), -1.0, 1e-5);
}

TEST(AlignBallGoal_Defensivterm) {
	// Auto zwischen eigenem Tor und Ball: defensiv ausgerichtet
	auto state = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, -3000, GOAL_Z)) }, Vec(0, 0, GOAL_Z));
	AlignBallGoalReward defOnly(0.f, 1.f);
	CHECK_NEAR(Single(&defOnly, state), 1.0, 1e-5);
}

// --- Geschwindigkeit zum Ball -------------------------------------------

TEST(VelocityPlayerToBall_volle_Geschwindigkeit) {
	// Auto und Ball auf gleicher Höhe: Richtung ist exakt +Y
	auto state = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, 93), Vec(0, CommonValues::CAR_MAX_SPEED, 0)) },
	                       Vec(0, 1000, 93));
	VelocityPlayerToBallReward r;
	CHECK_NEAR(Single(&r, state), 1.0, 1e-5);

	// Weg vom Ball -> -1
	auto away = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, 93), Vec(0, -CommonValues::CAR_MAX_SPEED, 0)) },
	                      Vec(0, 1000, 93));
	CHECK_NEAR(Single(&r, away), -1.0, 1e-5);
}

// --- Touch Ball-to-Goal Acceleration ------------------------------------

TEST(TouchBallToGoalAccel_nur_bei_eigenem_Kontakt) {
	TouchBallToGoalAccelReward r;
	ActionSet acts(1);

	auto s0 = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, 17)) }, Vec(0, 0, 93), Vec(0, 0, 0));
	r.Reset(s0);
	r.PreStep(s0);
	CHECK_NEAR(r.GetAllRewards(s0, acts, false)[0], 0.0, 1e-9);

	// Ball bekommt 1200 uu/s Richtung oranges Tor, Spieler hat berührt
	auto s1 = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, 17)) }, Vec(0, 0, 93), Vec(0, 1200, 0));
	s1.players[0].ballTouchedStep = true;
	r.PreStep(s1);
	float expected = 1200.f / CommonValues::BALL_MAX_SPEED;
	CHECK_NEAR(r.GetAllRewards(s1, acts, false)[0], expected, 1e-3);

	// Gleiche Ballgeschwindigkeit, aber kein Kontakt -> 0
	auto s2 = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, 17)) }, Vec(0, 100, 93), Vec(0, 1200, 0));
	r.PreStep(s2);
	CHECK_NEAR(r.GetAllRewards(s2, acts, false)[0], 0.0, 1e-9);
}

TEST(TouchBallToGoalAccel_Abbremsen_ist_negativ) {
	TouchBallToGoalAccelReward r;
	ActionSet acts(1);

	auto s0 = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, 17)) }, Vec(0, 0, 93), Vec(0, 2000, 0));
	r.Reset(s0);
	r.PreStep(s0);
	r.GetAllRewards(s0, acts, false);

	auto s1 = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, 17)) }, Vec(0, 0, 93), Vec(0, 500, 0));
	s1.players[0].ballTouchedStep = true;
	r.PreStep(s1);
	CHECK_NEAR(r.GetAllRewards(s1, acts, false)[0], -1500.f / CommonValues::BALL_MAX_SPEED, 1e-3);
}

// --- Einfache Rewards ----------------------------------------------------

TEST(SaveBoost_ist_Wurzel) {
	auto state = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, 17), {}, 0.64f) }, Vec(0, 500, 93));
	SaveBoostReward r(0.5f);
	CHECK_NEAR(Single(&r, state), 0.8, 1e-4);
}

TEST(InAir_nur_ohne_Bodenkontakt) {
	auto state = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, 500)) }, Vec(0, 500, 93));
	InAirReward r;
	CHECK_NEAR(Single(&r, state), 0.0, 1e-9);   // isOnGround ist in MakePlayer true

	state.players[0].carState.isOnGround = false;
	CHECK_NEAR(Single(&r, state), 1.0, 1e-9);

	state.players[0].carState.isDemoed = true;  // demoliert zählt nicht
	CHECK_NEAR(Single(&r, state), 0.0, 1e-9);
}

// --- Zusammenbau ---------------------------------------------------------

TEST(OffensivePotential_ist_KRC_der_drei_Komponenten) {
	auto state = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, GOAL_Z), Vec(0, CommonValues::CAR_MAX_SPEED, 0)) },
	                       Vec(0, 1000, GOAL_Z));
	RewardFunction* op = MakeOffensivePotential();

	// Komponenten von Hand: align = 1, vel = 1, dist = exp(-0.5*d/2300)
	float dist = 1000.f - CommonValues::BALL_RADIUS;
	float distReward = ParamDistanceReward::Compute(dist, CommonValues::CAR_MAX_SPEED, 1.f, 1.f);
	float expected = std::pow(1.0 * 1.0 * distReward, 1.0 / 3.0);

	CHECK_NEAR(Single(op, state), expected, 1e-3);
	delete op;
}

TEST(BuildLucyReward_gewichtet_Tore) {
	RewardWeights w = {};
	w.goal = 10.f; w.concede = 10.f;
	w.touchBallToGoalAccel = 0; w.offensivePotential = 0; w.distWeightedAlign = 0;
	w.velocityPlayerToBall = 0; w.saveBoost = 0; w.inAir = 0;

	RewardFunction* fn = BuildLucyReward(w);
	ActionSet acts(2);

	auto s0 = MakeState({ MakePlayer(1, Team::BLUE, Vec(0, 0, 17)),
	                      MakePlayer(2, Team::ORANGE, Vec(0, 100, 17)) }, Vec(0, 500, 93));
	fn->Reset(s0);
	fn->PreStep(s0);
	fn->GetAllRewards(s0, acts, false);

	// Blau trifft
	auto s1 = s0;
	s1.scoreLine[(int)Team::BLUE] = 1;
	fn->PreStep(s1);
	auto rewards = fn->GetAllRewards(s1, acts, false);
	CHECK_NEAR(rewards[0], 10.0, 1e-4);    // Torschütze-Team
	CHECK_NEAR(rewards[1], -10.0, 1e-4);   // Gegentor
	delete fn;
}

TEST(TeamSpirit_macht_Reward_zero_sum) {
	RewardWeights w = {};
	w.goal = 0; w.concede = 0; w.touchBallToGoalAccel = 0; w.offensivePotential = 0;
	w.distWeightedAlign = 0; w.saveBoost = 1.f; w.inAir = 0; w.velocityPlayerToBall = 0;
	w.teamSpirit = 1.f;

	RewardFunction* fn = BuildLucyReward(w);
	// 2v2, unterschiedliche Boost-Stände
	auto state = MakeState({
		MakePlayer(1, Team::BLUE, Vec(-500, 0, 17), {}, 1.00f),
		MakePlayer(2, Team::BLUE, Vec(500, 0, 17), {}, 0.25f),
		MakePlayer(3, Team::ORANGE, Vec(-500, 1000, 17), {}, 0.49f),
		MakePlayer(4, Team::ORANGE, Vec(500, 1000, 17), {}, 0.81f),
	}, Vec(0, 500, 93));

	ActionSet acts(4);
	fn->Reset(state);
	fn->PreStep(state);
	auto rewards = fn->GetAllRewards(state, acts, false);

	float sum = 0;
	for (float r : rewards) sum += r;
	CHECK_NEAR(sum, 0.0, 1e-4);
	// Bei teamSpirit=1 bekommen Teamkollegen denselben Wert
	CHECK_NEAR(rewards[0], rewards[1], 1e-5);
	CHECK_NEAR(rewards[2], rewards[3], 1e-5);
	delete fn;
}
