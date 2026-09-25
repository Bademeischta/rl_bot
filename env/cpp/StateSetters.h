// Mechanik-Curriculum über State-Setter (Bauplan v2 §9, PRISM-Punkt 5).
//
// Ein WeightedStateSetter wählt pro Episode nach Gewichten eine Szene aus. Die Gewichte
// kommen aus der Trainings-Config und sind damit gate-gesteuert veränderbar.
//
// Zufall (Audit H6): Mit Seed zieht jeder Setter aus einem eigenen std::mt19937_64 statt aus
// RocketSims zeitgeseedetem, thread-lokalem Engine. Damit sind Szenenwahl und Szenenparameter
// bei gleichem Seed reproduzierbar. Ohne Seed (Standardkonstruktor) bleibt das alte Verhalten.
// Nicht seedbar bleiben die Upstream-Teile: RandomState (rlgym-sim) und Arena::ResetToRandomKickoff
// (Kickoff-Spawn-Auswahl), beide ziehen weiter aus dem globalen Engine.
#pragma once

#include <RLGymSim_CPP/Utils/StateSetters/StateSetter.h>

#include <random>

namespace RLbot {
using namespace RLGSC;

typedef std::mt19937_64 EnvRng;

// SplitMix64-Mischung, damit benachbarte Seeds (envIndex 0, 1, 2 ...) unkorrelierte Ströme geben.
uint64_t MixSeed(uint64_t seed);
// Seed für Environment envIndex und Strom stream (0 = State-Setter, 1 = Obs-Shuffle, ...).
uint64_t SeedForEnv(int randomSeed, int envIndex, int stream);

// Gemeinsame Basis der eigenen Szenen: optionaler RNG, sonst RocketSims globaler.
class SceneSetter : public StateSetter {
public:
	EnvRng* rng = nullptr;

	float RandF(float min, float max) const;
	int RandSign() const;
};

// Kickoff wie im Spiel (nutzt RocketSims eigene Kickoff-Positionen; Spawn-Auswahl nicht seedbar).
class KickoffSetter : public SceneSetter {
public:
	virtual GameState ResetState(Arena* arena);
};

// Ball hoch in der Luft mit Geschwindigkeit, Autos am Boden mit viel Boost.
class AerialSetter : public SceneSetter {
public:
	virtual GameState ResetState(Arena* arena);
};

// Ball knapp über dem Boden und langsam, ein Auto direkt dahinter: Dribbling/Carry.
class DribbleSetter : public SceneSetter {
public:
	virtual GameState ResetState(Arena* arena);
};

// Ball an der Seitenwand in Höhe, Autos in der Nähe: Wall-Play.
class WallPlaySetter : public SceneSetter {
public:
	virtual GameState ResetState(Arena* arena);
};

// Autos in der Luft mit zufälliger Rotation und Drehrate: Recovery/Landung.
class RecoverySetter : public SceneSetter {
public:
	virtual GameState ResetState(Arena* arena);
};

// Ball fliegt aufs eigene Tor, Verteidiger dazwischen: Save/Defense.
class DefenseSetter : public SceneSetter {
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

	// Seed, mit dem der eigene RNG initialisiert wurde; -1 = ungeseedet (globaler Engine)
	int64_t seed = -1;

	// Ungeseedet: Szenenwahl und Szenenparameter aus RocketSims globalem Engine (altes Verhalten).
	explicit WeightedStateSetter(const StateSetterWeights& w);
	// Geseedet: eigener RNG, an alle eigenen Szenen durchgereicht.
	WeightedStateSetter(const StateSetterWeights& w, uint64_t seed);
	RG_NO_COPY(WeightedStateSetter);
	~WeightedStateSetter();

	virtual GameState ResetState(Arena* arena);

private:
	EnvRng rng;
	bool seeded = false;
	void Build(const StateSetterWeights& w);
};

} // namespace RLbot
