#include "StateSetters.h"

#include <RLGymSim_CPP/Math.h>
#include <RLGymSim_CPP/Utils/StateSetters/RandomState.h>

#include <algorithm>

namespace RLbot {

using RLGSC::CommonValues::BALL_RADIUS;

uint64_t MixSeed(uint64_t z) {
	// SplitMix64 (Steele, Lea, Flood 2014)
	z += 0x9E3779B97F4A7C15ull;
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

uint64_t SeedForEnv(int randomSeed, int envIndex, int stream) {
	uint64_t base = (uint64_t)(int64_t)randomSeed * 1000003ull;
	// 63 Bit, damit der Seed auch als int64_t nicht negativ ist (-1 ist der "ungeseedet"-Marker)
	return MixSeed(base + (uint64_t)envIndex * 7919ull + (uint64_t)stream * 104729ull) >> 1;
}

float SceneSetter::RandF(float min, float max) const {
	if (rng)
		return std::uniform_real_distribution<float>(min, max)(*rng);
	return ::Math::RandFloat(min, max);
}

int SceneSetter::RandSign() const {
	return RandF(0, 1) > 0.5f ? 1 : -1;
}

// Autos in fester Reihenfolge (Car-ID). arena->_cars ist ein std::unordered_set<Car*>: Die
// Reihenfolge hängt von den Speicheradressen ab, damit verteilte ein geseedeter Setter seine
// Zufallszahlen bei jedem Lauf anders auf die Autos (Nebenbefund R19 zu Audit H6).
static std::vector<Car*> CarsById(Arena* arena) {
	std::vector<Car*> cars(arena->_cars.begin(), arena->_cars.end());
	std::sort(cars.begin(), cars.end(), [](const Car* a, const Car* b) { return a->id < b->id; });
	return cars;
}

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

	for (Car* car : CarsById(arena)) {
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

	// Ballführer zufällig aus dem (geseedeten) RNG; vorher war es das erste Auto in
	// Hash-Reihenfolge, also adressabhängig. Die Verteilung bleibt gleich (jedes Auto gleich oft).
	auto cars = CarsById(arena);
	int carrier = std::min((int)cars.size() - 1, (int)RandF(0, (float)cars.size()));
	for (int i = 0; i < (int)cars.size(); i++) {
		Car* car = cars[i];
		if (i == carrier) {
			// Direkt hinter dem Ball, gleiche Richtung: Carry-Start
			Vec offsetDir = ballVel.Length() > 50 ? ballVel.Normalized() : Vec(0, 1, 0);
			CarState cs = {};
			cs.pos = ballPos - offsetDir * 120.f;
			cs.pos.z = 17;
			cs.vel = ballVel;
			cs.rotMat = Angle(std::atan2(offsetDir.y, offsetDir.x), 0, 0).ToRotMat();
			cs.boost = RandF(30, 100);
			car->SetState(cs);
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

	for (Car* car : CarsById(arena)) {
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

	for (Car* car : CarsById(arena)) {
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

	for (Car* car : CarsById(arena)) {
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
	Build(w);
}

WeightedStateSetter::WeightedStateSetter(const StateSetterWeights& w, uint64_t seed)
	: seed((int64_t)seed), rng(seed), seeded(true) {
	Build(w);
}

void WeightedStateSetter::Build(const StateSetterWeights& w) {
	auto add = [&](const char* name, float weight, StateSetter* setter) {
		if (weight > 0) {
			if (seeded)
				if (auto* scene = dynamic_cast<SceneSetter*>(setter))
					scene->rng = &rng;
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
	float roll = seeded
		? std::uniform_real_distribution<float>(0.f, totalWeight)(rng)
		: ::Math::RandFloat() * totalWeight;
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
