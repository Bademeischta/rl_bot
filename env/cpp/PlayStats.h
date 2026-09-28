// Spielanalyse aus der Zustandsfolge eines Spiels: Anstöße, Angriffsdrittel, Ballkontakte, Schüsse,
// Tore und Teamverhalten. Gemeinsame Grundlage für die Trainingsmetriken (train/cpp/Metrics.cpp)
// und die Kennzahlen je Seite im Duell (eval/cpp/duel.cpp), damit beide dasselbe messen.
//
// Eingabe ist pro Schritt der GameState aus Gym::Step (Snapshot 1 Tick nach der Aktion, wie im
// Training) und das done-Flag. Der erste Aufruf und jeder Aufruf nach done beginnen eine Episode.
// Ergebnisse, die sich über mehrere Schritte erstrecken (Anstoß, Aufenthalt im Angriffsdrittel),
// werden ausgegeben, sobald sie feststehen, spätestens am Episodenende oder mit Flush().
//
// Teams: 0 = Blau (greift das orange Tor bei +y an), 1 = Orange.
#pragma once

#include <RLGymSim_CPP/Utils/Gamestates/GameState.h>

#include <unordered_map>
#include <vector>

namespace RLbot {
using namespace RLGSC;

// Angriffsdrittel: |y| jenseits eines Drittels der Feldlänge (Ball in Blaus Angriffsdrittel bei y > 1706,7).
constexpr float OFF_THIRD_Y = CommonValues::BACK_WALL_Y / 3.f;
// Ein Aufenthalt im Angriffsdrittel endet erst, wenn der Ball so lange draußen war (Abpraller an der
// Drittellinie sollen keinen neuen Aufenthalt beginnen).
constexpr float OFF_THIRD_EXIT_SECS = 1.f;
// Ballhöhe, ab der eine Luftberührung als Aerial zählt: über dem, was ein Einfachsprung erreicht.
constexpr float AERIAL_TOUCH_MIN_HEIGHT = 450.f;
// Anstoß-Auswertung: Ballbesitz so lange nach der ersten Berührung, Tore bis so lange nach dem Anstoß.
constexpr float KICKOFF_POSSESSION_SECS = 3.f;
constexpr float KICKOFF_GOAL_SECS = 10.f;
// Team: "fährt zum Ball" = näher als COMMIT_DIST und Tempo Richtung Ball über COMMIT_SPEED;
// "sichert ab" = mindestens BACK_MARGIN näher am eigenen Tor (in y) als der Ball.
constexpr float COMMIT_DIST = 1500.f, COMMIT_SPEED = 500.f, BACK_MARGIN = 500.f;

struct KickoffResult {
	int firstTeam = -1;              // Team mit der ersten Berührung, -1 = keine in KICKOFF_GOAL_SECS
	float timeToTouch = 0;           // s vom Anstoß bis zur ersten Berührung (tickgenau)
	float touchSpeed = 0;            // Tempo des ersten Berührers im Schritt vor der Berührung (uu/s)
	float loserSpeed = 0;            // Tempo des schnellsten Gegners im selben Moment
	float boostUsed[2] = { 0, 0 };   // Boost-Verbrauch je Team bis zur ersten Berührung (Mittel je Spieler, 0-100)
	int ballHalfTeam = -1;           // KICKOFF_POSSESSION_SECS nach der Berührung: in wessen Angriffshälfte der Ball ist
	int closerTeam = -1;             // ... und wessen Spieler dem Ball am nächsten ist
	int goalTeam = -1;               // Tor innerhalb von KICKOFF_GOAL_SECS nach dem Anstoß (-1 = keins)
};

struct OffenseSpell {
	int team = -1;                   // angreifendes Team
	float seconds = 0;               // Dauer des Aufenthalts im Angriffsdrittel
	bool goal = false;               // endete mit einem Tor dieses Teams
};

struct TouchEvent {
	int team = -1;
	uint32_t carId = 0;
	bool carInAir = false;           // Auto beim Snapshot nicht auf dem Boden
	float ballHeight = 0;            // Ballhöhe beim Kontakt (aus BallHitInfo)
	uint64_t tick = 0;               // Arena-Tick des Kontakts
};

struct TeamSample {
	int team = -1;
	float mateDist = 0;              // mittlerer Abstand der Mitspieler untereinander
	bool doubleCommit = false;       // mindestens zwei fahren gleichzeitig zum Ball
	bool lastBack = false;           // mindestens einer sichert hinter dem Ball ab
};

struct PlayEvents {
	std::vector<KickoffResult> kickoffs;
	std::vector<OffenseSpell> spells;
	std::vector<TouchEvent> touches;
	std::vector<TeamSample> teams;
	int shots[2] = { 0, 0 };         // neue Schüsse je Team (RocketSims Schuss-Ereignis)
	int goalTeam = -1;               // Tor in diesem Schritt
	int ballThirdTeam = -1;          // Ball im Angriffsdrittel dieses Teams, -1 = Mittelfeld
	int players = 0;
	int playersPerTeam[2] = { 0, 0 };
};

class PlayTracker {
public:
	explicit PlayTracker(int tickSkip = 8) : tickSkip(tickSkip) {}

	PlayEvents Step(const GameState& state, bool done);
	// Offene Auswertungen (Anstoß, Aufenthalt im Drittel) sofort abschließen, z. B. am Spielende im
	// Duell, wo kein done kommt. Danach beginnt der nächste Step eine neue Episode.
	PlayEvents Flush();

	static int TeamOf(const PlayerData& p) { return p.team == Team::BLUE ? 0 : 1; }
	// Team, in dessen Angriffsdrittel diese y-Position liegt (-1 = Mittelfeld)
	static int AttackingThird(float y);

private:
	int tickSkip;
	bool newEpisode = true;
	int64_t step = 0;                // Schritt innerhalb der Episode, 1 = erster Schritt
	uint64_t resetTick = 0;

	// Anstoß
	bool kickoffOpen = false;
	int64_t kickoffTouchStep = -1;
	KickoffResult kickoff;

	// Angriffsdrittel
	bool spellOpen = false;
	OffenseSpell spell;
	int64_t spellStart = 0, spellLastIn = 0;

	// Je Auto: Schüsse, Boost und Tempo im vorigen Schritt
	std::unordered_map<uint32_t, int> lastShots;
	std::unordered_map<uint32_t, float> lastBoost, lastSpeed;

	float Seconds(int64_t steps) const { return steps * tickSkip / 120.f; }
	void BeginEpisode(const GameState& state);
	void CloseKickoff(PlayEvents& ev);
	void CloseSpell(PlayEvents& ev, bool goal);
};

} // namespace RLbot
