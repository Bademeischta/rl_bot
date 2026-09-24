#include "StateSetters.h"

#include <RLGymSim_CPP/Math.h>
#include <RLGymSim_CPP/Utils/StateSetters/RandomState.h>

namespace RLbot {

using RLGSC::CommonValues::BALL_RADIUS;

static float RandF(float min, float max) { return ::Math::RandFloat(min, max); }
static int RandSign() { return ::Math::RandFloat() > 0.5f ? 1 : -1; }

// Autos sauber am Boden absetzen, mit Blick auf einen Zielpunkt.
static void PlaceOnGround(Car* car, Vec pos, Vec lookAt, float boost) {
	CarState cs = {};
	cs.pos = Vec(pos.x, pos.y, 17);
	Vec dir = (lookAt - cs.pos);
	dir.z = 0;
	float yaw = std::atan2(dir.y, dir.x);
	cs.rotMat = Angle(yaw, 0, 0).ToRotMat();
	cs.boost = boost;
	car->SetState(cs);
}

GameState KickoffSetter::ResetState(Arena* arena) {
	arena->ResetToRandomKickoff();
	return GameState(arena);
}

GameState AerialSetter::ResetState(Arena* arena) {
	arena->ResetToRandomKickoff();

	BallState bs = {};
	bs.pos = Vec(RandF(-2500, 2500), RandF(-3500, 3500), RandF(700, 1700));
	bs.vel = Vec(RandF(-600, 600), RandF(-600, 600), RandF(-200, 700));
	arena->ball->SetState(bs);

	for (Car* car : arena->_cars) {
		Vec pos = bs.pos + Vec(RandF(-1800, 1800), RandF(-1800, 1800), 0);
		pos.x = RS_CLAMP(pos.x, -3800.f, 3800.f);
		pos.y = RS_CLAMP(pos.y, -4800.f, 4800.f);
		PlaceOnGround(car, pos, bs.pos, RandF(60, 100));
	}
	return GameState(arena);
}

GameState DribbleSetter::ResetState(Arena* arena) {
	arena->ResetToRandomKickoff();

	Vec ballPos = Vec(RandF(-2500, 2500), RandF(-3000, 3000), BALL_RADIUS + RandF(0, 40));
	Vec ballVel = Vec(RandF(-300, 300), RandF(-300, 300), 0);

	BallState bs = {};
	bs.pos = ballPos;
	bs.vel = ballVel;
	arena->ball->SetState(bs);

	bool first = true;
	for (Car* car : arena->_cars) {
		if (first) {
			// Direkt hinter dem Ball, gleiche Richtung: Carry-Start
			Vec offsetDir = ballVel.Length() > 50 ? ballVel.Normalized() : Vec(0, 1, 0);
			CarState cs = {};
			cs.pos = ballPos - offsetDir * 120.f;
			cs.pos.z = 17;
			cs.vel = ballVel;
			cs.rotMat = Angle(std::atan2(offsetDir.y, offsetDir.x), 0, 0).ToRotMat();
			cs.boost = RandF(30, 100);
			car->SetState(cs);
			first = false;
		} else {
			Vec pos = Vec(RandF(-3500, 3500), RandF(-4500, 4500), 17);
			PlaceOnGround(car, pos, ballPos, RandF(20, 100));
		}
	}
	return GameState(arena);
}

GameState WallPlaySetter::ResetState(Arena* arena) {
	arena->ResetToRandomKickoff();

	float side = (float)RandSign();
	BallState bs = {};
	bs.pos = Vec(side * RandF(3400, 3900), RandF(-3500, 3500), RandF(300, 1400));
	bs.vel = Vec(side * RandF(0, 500), RandF(-500, 500), RandF(-300, 300));
	arena->ball->SetState(bs);

	for (Car* car : arena->_cars) {
		Vec pos = Vec(side * RandF(2200, 3600), bs.pos.y + RandF(-1500, 1500), 17);
		pos.y = RS_CLAMP(pos.y, -4800.f, 4800.f);
		PlaceOnGround(car, pos, bs.pos, RandF(40, 100));
	}
	return GameState(arena);
}

GameState RecoverySetter::ResetState(Arena* arena) {
	arena->ResetToRandomKickoff();

	BallState bs = {};
	bs.pos = Vec(RandF(-2500, 2500), RandF(-3500, 3500), RandF(BALL_RADIUS, 1200));
	bs.vel = Vec(RandF(-800, 800), RandF(-800, 800), RandF(-200, 400));
	arena->ball->SetState(bs);

	for (Car* car : arena->_cars) {
		CarState cs = {};
		cs.pos = Vec(RandF(-3500, 3500), RandF(-4500, 4500), RandF(300, 1600));
		cs.vel = Vec(RandF(-1200, 1200), RandF(-1200, 1200), RandF(-600, 600));
		cs.angVel = Vec(RandF(-4.5f, 4.5f), RandF(-4.5f, 4.5f), RandF(-4.5f, 4.5f));
		cs.rotMat = Angle(RandF(-M_PI, M_PI), RandF(-M_PI / 2, M_PI / 2), RandF(-M_PI, M_PI)).ToRotMat();
		cs.boost = RandF(0, 60);
		car->SetState(cs);
	}
	return GameState(arena);
}

GameState DefenseSetter::ResetState(Arena* arena) {
	arena->ResetToRandomKickoff();

	// Ball fliegt auf das blaue Tor; blaue Autos verteidigen, orange greifen an.
	BallState bs = {};
	bs.pos = Vec(RandF(-2000, 2000), RandF(-1500, 1500), RandF(BALL_RADIUS, 900));
	Vec target = Vec(RandF(-700, 700), -CommonValues::BACK_WALL_Y, RandF(100, 600));
	bs.vel = (target - bs.pos).Normalized() * RandF(1200, 2600);
	arena->ball->SetState(bs);

	for (Car* car : arena->_cars) {
		if (car->team == Team::BLUE) {
			Vec pos = Vec(RandF(-1200, 1200), RandF(-4600, -3000), 17);
			PlaceOnGround(car, pos, bs.pos, RandF(20, 80));
		} else {
			Vec pos = bs.pos + Vec(RandF(-1200, 1200), RandF(200, 1800), 0);
			pos.x = RS_CLAMP(pos.x, -3800.f, 3800.f);
			pos.y = RS_CLAMP(pos.y, -4800.f, 4800.f);
			PlaceOnGround(car, pos, bs.pos, RandF(20, 100));
		}
	}
	return GameState(arena);
}

WeightedStateSetter::WeightedStateSetter(const StateSetterWeights& w) {
	auto add = [&](const char* name, float weight, StateSetter* setter) {
		if (weight > 0) {
			names.push_back(name);
			weights.push_back(weight);
			setters.push_back(setter);
			totalWeight += weight;
		} else {
			delete setter;
		}
	};
	add("kickoff", w.kickoff, new KickoffSetter());
	add("random", w.random, new RandomState(true, true, true));
	add("aerial", w.aerial, new AerialSetter());
	add("dribble", w.dribble, new DribbleSetter());
	add("wall_play", w.wallPlay, new WallPlaySetter());
	add("recovery", w.recovery, new RecoverySetter());
	add("defense", w.defense, new DefenseSetter());

	if (setters.empty())
		RG_ERR_CLOSE("WeightedStateSetter: alle Gewichte sind 0");
}

WeightedStateSetter::~WeightedStateSetter() {
	for (auto s : setters)
		delete s;
}

GameState WeightedStateSetter::ResetState(Arena* arena) {
	float roll = ::Math::RandFloat() * totalWeight;
	int picked = (int)setters.size() - 1;
	for (int i = 0; i < (int)setters.size(); i++) {
		roll -= weights[i];
		if (roll <= 0) {
			picked = i;
			break;
		}
	}
	lastPicked = picked;
	return setters[picked]->ResetState(arena);
}

} // namespace RLbot
