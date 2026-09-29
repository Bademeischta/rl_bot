// Aktionsauswahl aus den Policy-Wahrscheinlichkeiten, wie sie im Duell und (Python-Gegenstück
// deploy/action_select.py) im Bot möglich ist.
//
// Die 90er-Aktionstabelle (RLGymSim DiscreteAction) enthält viele Einträge, die in einer Lage
// dasselbe bewirken. Am Boden ohne Sprung wirken nur Gas, Lenken, Boost und Handbremse: Für
// "Gas + Boost geradeaus" gibt es 9 Einträge (8 Luftaktionen mit Nicken/Rollen plus den
// Bodeneintrag), für "Gas ohne Boost geradeaus" nur einen. argmax über Einträge wählt deshalb
// systematisch seltener Boost, als die Policy es will. argmax über Wirkungsklassen summiert erst
// die Wahrscheinlichkeiten gleichwirkender Einträge.
//
// Wirkungsklassen:
//   am Boden, ohne Sprung: (Gas, Lenken, Boost, Handbremse); Nicken/Rollen wirken am Boden nicht
//   am Boden, mit Sprung:  jeder Eintrag einzeln (nach dem Abheben wirken Nicken/Gieren/Rollen
//                          noch im selben Schritt)
//   in der Luft:           (Nicken, Gieren, Rollen, Sprung, Boost); Gas beschleunigt in der Luft
//                          nur um ~4 uu/s je Schritt, die Handbremse wirkt nicht
#pragma once

#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>

#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace RLbot {

enum class SelectMode { SAMPLE, ARGMAX, ARGMAX_GROUP };

inline bool ParseSelectMode(const std::string& s, SelectMode& out) {
	if (s == "sample") out = SelectMode::SAMPLE;
	else if (s == "argmax") out = SelectMode::ARGMAX;
	else if (s == "argmax_group") out = SelectMode::ARGMAX_GROUP;
	else return false;
	return true;
}

inline const char* SelectModeName(SelectMode m) {
	return m == SelectMode::SAMPLE ? "sample" : (m == SelectMode::ARGMAX ? "argmax" : "argmax_group");
}

// Klassennummer je Tabelleneintrag (gleiche Nummer = gleiche Wirkung in dieser Lage).
inline std::vector<int> EffectGroups(const std::vector<RLGSC::Action>& table, bool onGround) {
	std::map<std::vector<float>, int> ids;
	std::vector<int> out;
	out.reserve(table.size());
	for (size_t i = 0; i < table.size(); i++) {
		const auto& a = table[i];
		// Reihenfolge der Elemente: throttle, steer, pitch, yaw, roll, jump, boost, handbrake
		std::vector<float> key;
		bool jump = a[5] >= 0.5f;
		if (onGround && !jump)
			key = { 0.f, a[0], a[1], a[6], a[7] };
		else if (onGround)
			key = { 1.f, (float)i };
		else
			key = { 2.f, a[2], a[3], a[4], a[5], a[6] };
		auto it = ids.find(key);
		if (it == ids.end())
			it = ids.emplace(key, (int)ids.size()).first;
		out.push_back(it->second);
	}
	return out;
}

// argmax über Wirkungsklassen: Klasse mit der größten Summe, darin der wahrscheinlichste Eintrag.
inline int ArgmaxGroup(const float* probs, const std::vector<int>& groups) {
	std::map<int, float> sums;
	for (size_t i = 0; i < groups.size(); i++)
		sums[groups[i]] += probs[i];
	int bestGroup = -1;
	float best = -1;
	for (auto& [g, p] : sums)
		if (p > best) {
			best = p;
			bestGroup = g;
		}
	int bestIdx = -1;
	for (size_t i = 0; i < groups.size(); i++)
		if (groups[i] == bestGroup && (bestIdx < 0 || probs[i] > probs[bestIdx]))
			bestIdx = (int)i;
	return bestIdx;
}

} // namespace RLbot
