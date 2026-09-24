// Mechanik-Curriculum über State-Setter (Bauplan v2 §9, PRISM-Punkt 5).
//
// Ein WeightedStateSetter wählt pro Episode nach Gewichten eine Szene aus. Die Gewichte
// kommen aus der Trainings-Config und sind damit gate-gesteuert veränderbar.
#pragma once

#include <RLGymSim_CPP/Utils/StateSetters/StateSetter.h>

namespace RLbot {
using namespace RLGSC;

// Kickoff wie im Spiel (nutzt RocketSims eigene Kickoff-Positionen).
class KickoffSetter : public StateSetter {
public:
	virtual GameState ResetState(Arena* arena);
};

// Ball hoch in der Luft mit Geschwindigkeit, Autos am Boden mit viel Boost.
class AerialSetter : public StateSetter {
public:
	virtual GameState ResetState(Arena* arena);
};

// Ball knapp über dem Boden und langsam, ein Auto direkt dahinter: Dribbling/Carry.
class DribbleSetter : public StateSetter {
public:
	virtual GameState ResetState(Arena* arena);
};

// Ball an der Seitenwand in Höhe, Autos in der Nähe: Wall-Play.
class WallPlaySetter : public StateSetter {
public:
	virtual GameState ResetState(Arena* arena);
};

// Autos in der Luft mit zufälliger Rotation und Drehrate: Recovery/Landung.
class RecoverySetter : public StateSetter {
public:
	virtual GameState ResetState(Arena* arena);
};

// Ball fliegt aufs eigene Tor, Verteidiger dazwischen: Save/Defense.
class DefenseSetter : public StateSetter {
public:
	virtual GameState ResetState(Arena* arena);
};

// Gewichte der Szenen. Alles 0 außer kickoff = reines Kickoff-Training.
struct StateSetterWeights {
	float kickoff = 1.f;
	float random = 1.f;
	float aerial = 0.f;
	float dribble = 0.f;
	float wallPlay = 0.f;
	float recovery = 0.f;
	float defense = 0.f;
};

// Wählt pro Episode eine Szene nach Gewichten aus.
class WeightedStateSetter : public StateSetter {
public:
	std::vector<StateSetter*> setters;
	std::vector<float> weights;
	std::vector<std::string> names;
	float totalWeight = 0;

	// Index der zuletzt gewählten Szene (für Metriken)
	int lastPicked = -1;

	explicit WeightedStateSetter(const StateSetterWeights& w);
	RG_NO_COPY(WeightedStateSetter);
	~WeightedStateSetter();

	virtual GameState ResetState(Arena* arena);
};

} // namespace RLbot
