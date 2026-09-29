// Wirkungsklassen der Aktionstabelle und argmax über Klassen (env/cpp/ActionSelect.h), an der echten
// Tabelle aus RLGymSim DiscreteAction.
#include "test_util.h"

#include "env/cpp/ActionSelect.h"

#include <map>

using namespace RLbot;
using namespace RLGSC;

namespace {

int IndexOf(const std::vector<Action>& table, std::array<float, 8> want) {
	for (size_t i = 0; i < table.size(); i++) {
		bool same = true;
		for (int e = 0; e < 8; e++)
			same &= table[i][e] == want[e];
		if (same)
			return (int)i;
	}
	throw std::runtime_error("nicht in der Tabelle");
}

int GroupSize(const std::vector<int>& groups, int index) {
	int n = 0;
	for (int g : groups)
		n += g == groups[index];
	return n;
}

} // namespace

TEST(Aktionsauswahl_Boden_Gas_mit_Boost_hat_neun_gleichwirkende_Eintraege) {
	auto table = DiscreteAction().actions;
	CHECK_EQ(table.size(), (size_t)90);
	auto ground = EffectGroups(table, true);
	int boost = IndexOf(table, { 1, 0, 0, 0, 0, 0, 1, 0 });
	int noBoost = IndexOf(table, { 1, 0, 0, 0, 0, 0, 0, 0 });
	CHECK_EQ(GroupSize(ground, boost), 9);          // Bodeneintrag + 8 Luftaktionen mit Nicken/Rollen
	CHECK_EQ(GroupSize(ground, noBoost), 1);        // Gas ohne Boost gibt es nur einmal
	// Luftaktion (Nicken -1, Rollen -1, Boost) wirkt am Boden wie Gas + Boost geradeaus
	CHECK_EQ(ground[IndexOf(table, { 1, 0, -1, 0, -1, 0, 1, 0 })], ground[boost]);
	// Sprünge bleiben am Boden einzeln: nach dem Abheben wirken Nicken und Rollen noch im Schritt
	for (size_t i = 0; i < table.size(); i++)
		if (table[i][5] >= 0.5f)
			CHECK_EQ(GroupSize(ground, (int)i), 1);
}

TEST(Aktionsauswahl_Luft_ignoriert_Gas_und_Handbremse) {
	auto table = DiscreteAction().actions;
	auto air = EffectGroups(table, false);
	// Bodeneinträge ohne Boost geradeaus: Gas -1/0/1 und Handbremse 0/1 sind in der Luft gleich
	int coast = IndexOf(table, { 0, 0, 0, 0, 0, 0, 0, 0 });
	CHECK_EQ(GroupSize(air, coast), 6);
	// Luftaktionen unterscheiden sich in Nicken/Gieren/Rollen/Sprung/Boost: jede einzeln
	int flip = IndexOf(table, { 0, 0, -1, 0, 0, 1, 0, 1 });
	CHECK_EQ(GroupSize(air, flip), 1);
	std::map<int, int> sizes;
	for (int g : air) sizes[g]++;
	int total = 0;
	for (auto& [g, n] : sizes) total += n;
	CHECK_EQ(total, 90);
}

TEST(Aktionsauswahl_argmax_ueber_Klassen_waehlt_die_gewollte_Wirkung) {
	auto table = DiscreteAction().actions;
	auto ground = EffectGroups(table, true);
	int boost = IndexOf(table, { 1, 0, 0, 0, 0, 0, 1, 0 });
	int noBoost = IndexOf(table, { 1, 0, 0, 0, 0, 0, 0, 0 });
	// Boost insgesamt 0,72 auf 9 Einträge verteilt, ohne Boost 0,10 auf einem, Rest gleichmäßig
	std::vector<float> probs(90, 0.f);
	float rest = 1.f - 0.72f - 0.10f;
	int others = 0;
	for (int i = 0; i < 90; i++)
		if (ground[i] != ground[boost] && i != noBoost) others++;
	for (int i = 0; i < 90; i++)
		probs[i] = ground[i] == ground[boost] ? 0.08f : (i == noBoost ? 0.10f : rest / others);
	int plain = 0;
	for (int i = 1; i < 90; i++)
		if (probs[i] > probs[plain]) plain = i;
	CHECK_EQ(plain, noBoost);                              // argmax über Einträge: kein Boost
	int grouped = ArgmaxGroup(probs.data(), ground);
	CHECK_EQ(ground[grouped], ground[boost]);              // argmax über Klassen: Boost
}

TEST(Aktionsauswahl_Modus_Namen) {
	SelectMode m;
	CHECK(ParseSelectMode("sample", m) && m == SelectMode::SAMPLE);
	CHECK(ParseSelectMode("argmax", m) && m == SelectMode::ARGMAX);
	CHECK(ParseSelectMode("argmax_group", m) && m == SelectMode::ARGMAX_GROUP);
	CHECK(!ParseSelectMode("greedy", m));
	CHECK_EQ(std::string(SelectModeName(SelectMode::ARGMAX_GROUP)), std::string("argmax_group"));
}
