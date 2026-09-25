// Prüft die Episoden-Ende-Metriken (Audit M8) und die Aggregation über Spiel-Reports (M1).
#include "test_util.h"

#include "env/cpp/Obs.h"
#include "env/cpp/StateSetters.h"
#include "env/cpp/TimeoutCondition.h"
#include "train/cpp/Metrics.h"

#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>
#include <RLGymSim_CPP/Utils/RewardFunctions/CommonRewards.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/NoTouchCondition.h>

using namespace RLGSC;
using namespace RLbot;
using RLGPC::Report;

static GameState StateWithBallAt(Vec ballPos) {
	GameState s = {};
	s.ball.pos = ballPos;
	s.ball.rotMat = RotMat::GetIdentity();
	s.ballInv = s.ball.Invert();
	PlayerData p = {};
	p.carId = 1;
	p.team = Team::BLUE;
	p.phys.rotMat = RotMat::GetIdentity();
	p.physInv = p.phys.Invert();
	p.carState.isOnGround = true;
	s.players.push_back(p);
	return s;
}

// Match ohne Arena: nur die Terminal-Bedingungen werden gebraucht.
struct TestMatch {
	NoTouchCondition* noTouch;
	TimeoutCondition* timeout;
	Match* match;

	TestMatch(int noTouchSteps, int timeoutSteps) {
		noTouch = new NoTouchCondition(noTouchSteps);
		timeout = new TimeoutCondition(timeoutSteps);
		match = new Match(new EventReward({}), { noTouch, timeout, new GoalScoreCondition() },
		                  new StackedPaddedOBS(3, 5, false), new DiscreteAction(), new KickoffSetter(), 1, true);
	}
	~TestMatch() { delete match; }
};

TEST(Metrik_Tor_wird_als_Tor_erkannt) {
	TestMatch tm(10, 100);
	auto state = StateWithBallAt(Vec(0, 6000, 93));   // hinter der Torlinie
	auto end = ClassifyEpisodeEnd(state, tm.match, false);
	CHECK(end.goal);
	CHECK(!end.noTouch);
	CHECK(!end.timeLimit);

	Report m;
	AccumEpisodeEnd(end, 321, "kickoff", m);
	CHECK_NEAR(m.GetAvg("ep_end_goal"), 1.0, 1e-9);
	CHECK_NEAR(m.GetAvg("ep_end_timeout"), 0.0, 1e-9);
	CHECK_NEAR(m.GetAvg("ep_end_notouch"), 0.0, 1e-9);
	CHECK_NEAR(m.GetAvg("ep_end_time"), 0.0, 1e-9);
	CHECK_NEAR(m.GetAvg("ep_length_steps"), 321.0, 1e-9);
	CHECK_NEAR(m.GetAvg("scene_kickoff_goal"), 1.0, 1e-9);
	CHECK_NEAR(m.GetAvg("scene_kickoff_length"), 321.0, 1e-9);
}

TEST(Metrik_NoTouch_Timeout_wird_erkannt) {
	TestMatch tm(3, 100);
	auto state = StateWithBallAt(Vec(0, 0, 93));
	tm.match->EpisodeReset(state);
	bool done = false;
	for (int i = 0; i < 3; i++)
		done = tm.match->IsDone(state);          // niemand berührt den Ball
	CHECK(done);

	auto end = ClassifyEpisodeEnd(state, tm.match, false);
	CHECK(!end.goal);
	CHECK(end.noTouch);
	CHECK(!end.timeLimit);

	Report m;
	AccumEpisodeEnd(end, 3, "", m);
	CHECK_NEAR(m.GetAvg("ep_end_goal"), 0.0, 1e-9);
	CHECK_NEAR(m.GetAvg("ep_end_timeout"), 1.0, 1e-9);
	CHECK_NEAR(m.GetAvg("ep_end_notouch"), 1.0, 1e-9);
	CHECK_NEAR(m.GetAvg("ep_end_time"), 0.0, 1e-9);
	CHECK(!m.Has("scene__goal_avg_total"));
}

TEST(Metrik_Spielzeit_Timeout_wird_erkannt) {
	TestMatch tm(100, 4);
	auto state = StateWithBallAt(Vec(0, 0, 93));
	state.players[0].ballTouchedStep = true;     // NoTouch darf nicht auslösen
	tm.match->EpisodeReset(state);
	bool done = false;
	for (int i = 0; i < 4; i++)
		done = tm.match->IsDone(state);
	CHECK(done);

	auto end = ClassifyEpisodeEnd(state, tm.match, false);
	CHECK(!end.goal);
	CHECK(!end.noTouch);
	CHECK(end.timeLimit);

	Report m;
	AccumEpisodeEnd(end, 4, "", m);
	CHECK_NEAR(m.GetAvg("ep_end_time"), 1.0, 1e-9);
	CHECK_NEAR(m.GetAvg("ep_end_timeout"), 1.0, 1e-9);
}

TEST(Metrik_Mittelwerte_ueber_mehrere_Episoden) {
	Report m;
	AccumEpisodeEnd(EpisodeEnd{ true, false, false, false }, 100, "kickoff", m);
	AccumEpisodeEnd(EpisodeEnd{ false, true, false, false }, 300, "kickoff", m);
	AccumEpisodeEnd(EpisodeEnd{ false, false, true, false }, 500, "aerial", m);
	CHECK_NEAR(m.GetAvg("ep_end_goal"), 1.0 / 3.0, 1e-9);
	CHECK_NEAR(m.GetAvg("ep_end_timeout"), 2.0 / 3.0, 1e-9);
	CHECK_NEAR(m.GetAvg("ep_end_notouch"), 1.0 / 3.0, 1e-9);
	CHECK_NEAR(m.GetAvg("ep_end_time"), 1.0 / 3.0, 1e-9);
	CHECK_NEAR(m.GetAvg("ep_length_steps"), 300.0, 1e-9);
	CHECK_NEAR(m.GetAvg("scene_kickoff_goal"), 0.5, 1e-9);
	CHECK_NEAR(m.GetAvg("scene_kickoff_length"), 200.0, 1e-9);
	CHECK_NEAR(m.GetAvg("scene_aerial_goal"), 0.0, 1e-9);
}

TEST(Metrik_Episodenlaenge_pro_Spiel_getrennt) {
	EpisodeLengthTracker tracker;
	int gameA = 0, gameB = 0;
	CHECK_EQ((int)tracker.OnEpisodeEnd(&gameA, 250), 250);   // erste Episode ab Step 0
	CHECK_EQ((int)tracker.OnEpisodeEnd(&gameB, 40), 40);
	CHECK_EQ((int)tracker.OnEpisodeEnd(&gameA, 600), 350);
	CHECK_EQ((int)tracker.OnEpisodeEnd(&gameB, 41), 1);
}

TEST(Metrik_Szenenname_kommt_vom_WeightedStateSetter) {
	StateSetterWeights w = {};
	w.kickoff = 1; w.random = 1;
	auto* setter = new WeightedStateSetter(w);
	Match match(new EventReward({}), {}, new StackedPaddedOBS(3, 5, false), new DiscreteAction(), setter, 1, true);

	CHECK_EQ(CurrentSceneName(&match), std::string(""));   // noch keine Episode gestartet
	setter->lastPicked = 1;
	CHECK_EQ(CurrentSceneName(&match), std::string("random"));
	CHECK_EQ(CurrentSceneName(nullptr), std::string(""));

	Match plain(new EventReward({}), {}, new StackedPaddedOBS(3, 5, false), new DiscreteAction(), new KickoffSetter(), 1, true);
	CHECK_EQ(CurrentSceneName(&plain), std::string(""));
}

TEST(Aggregation_mittelt_alle_AccumAvg_Schluessel_gewichtet) {
	// Spiel 1: 3 Steps ball_speed, 1 Episodenende; Spiel 2: 1 Step, Szene, die Spiel 1 nicht hat
	Report a, b;
	a.AccumAvg("ball_speed", 100); a.AccumAvg("ball_speed", 200); a.AccumAvg("ball_speed", 300);
	a.AccumAvg("ep_end_goal", 1);
	b.AccumAvg("ball_speed", 1000);
	b.AccumAvg("scene_aerial_goal", 0);
	b["Nicht-Avg-Schluessel"] = 42;

	Report out;
	AggregateGameMetrics({ a, b }, out);
	// gewichtet nach Step-Anzahl, nicht Mittel der Mittel: (600 + 1000) / 4
	CHECK_NEAR(out["ball_speed"], 400.0, 1e-9);
	CHECK_NEAR(out["ep_end_goal"], 1.0, 1e-9);
	CHECK_NEAR(out["scene_aerial_goal"], 0.0, 1e-9);
	CHECK(!out.Has("Nicht-Avg-Schluessel"));
	CHECK(!out.Has("ball_speed_avg_total"));
}

TEST(Aggregation_ohne_Daten_schreibt_nichts) {
	Report out;
	AggregateGameMetrics({}, out);
	CHECK(out.data.empty());
	Report empty;
	AggregateGameMetrics({ empty }, out);
	CHECK(out.data.empty());
}

// --- metrics.csv (Audit M1, N4) ----------------------------------------------

#include <filesystem>
#include <fstream>

static std::filesystem::path TempCSV(const char* name) {
	auto p = std::filesystem::temp_directory_path() / (std::string("rlbot_test_") + name + ".csv");
	std::filesystem::remove(p);
	return p;
}

static std::vector<std::string> ReadLines(const std::filesystem::path& p) {
	std::ifstream f(p);
	std::vector<std::string> lines;
	std::string line;
	while (std::getline(f, line)) lines.push_back(line);
	return lines;
}

TEST(CSV_neue_Datei_bekommt_genau_eine_Kopfzeile) {
	auto p = TempCSV("neu");
	MetricsCSVWriter w(p);
	Report r; r["Cumulative Timesteps"] = 100000; r["ball_speed"] = 1234.5;
	r["ball_speed_avg_total"] = 1; r["ball_speed_avg_count"] = 1;   // interne Schlüssel bleiben draußen
	w.Append(r);
	r["Cumulative Timesteps"] = 200000;
	w.Append(r);
	auto lines = ReadLines(p);
	CHECK_EQ((int)lines.size(), 3);
	CHECK_EQ(lines[0], std::string("\"Cumulative Timesteps\",\"ball_speed\""));
	CHECK_EQ(lines[1], std::string("100000,1234.5"));
	CHECK_EQ(lines[2], std::string("200000,1234.5"));
	std::filesystem::remove(p);
}

TEST(CSV_Fortsetzen_haengt_keine_zweite_Kopfzeile_an) {
	auto p = TempCSV("fortsetzen");
	{
		MetricsCSVWriter w(p);
		Report r; r["A"] = 1; r["B"] = 2;
		w.Append(r);
	}
	{
		MetricsCSVWriter w(p);   // neuer Prozess, gleiche Datei
		Report r; r["A"] = 3; r["B"] = 4;
		w.Append(r);
		CHECK_EQ((int)w.Columns().size(), 2);
	}
	auto lines = ReadLines(p);
	CHECK_EQ((int)lines.size(), 3);
	CHECK_EQ(lines[0], std::string("\"A\",\"B\""));
	CHECK_EQ(lines[1], std::string("1,2"));
	CHECK_EQ(lines[2], std::string("3,4"));
	std::filesystem::remove(p);
}

TEST(CSV_neuer_Schluessel_wird_hinten_angehaengt_und_Kopfzeile_neu_geschrieben) {
	auto p = TempCSV("neuer_schluessel");
	MetricsCSVWriter w(p);
	Report r1; r1["A"] = 1; r1["B"] = 2;
	w.Append(r1);
	Report r2; r2["A"] = 3; r2["B"] = 4; r2["Skill Rating 2v2"] = 1000;
	w.Append(r2);
	Report r3; r3["A"] = 5;   // B fehlt in dieser Iteration -> leeres Feld
	w.Append(r3);
	auto lines = ReadLines(p);
	CHECK_EQ((int)lines.size(), 4);
	CHECK_EQ(lines[0], std::string("\"A\",\"B\",\"Skill Rating 2v2\""));
	CHECK_EQ(lines[1], std::string("1,2"));            // alte Zeile bleibt unverändert (kürzer)
	CHECK_EQ(lines[2], std::string("3,4,1000"));
	CHECK_EQ(lines[3], std::string("5,,"));

	// Fortsetzen übernimmt die erweiterte Kopfzeile in genau dieser Reihenfolge
	MetricsCSVWriter w2(p);
	Report r4; r4["Skill Rating 2v2"] = 7; r4["B"] = 6; r4["A"] = 5;
	w2.Append(r4);
	CHECK_EQ(ReadLines(p).back(), std::string("5,6,7"));
	std::filesystem::remove(p);
}

TEST(CSV_nan_und_inf_werden_woertlich_geschrieben) {
	// Review-Befund R5: leer geschrieben erkannte check_abort.py ein NaN nie
	auto p = TempCSV("nan");
	MetricsCSVWriter w(p);
	Report r; r["A"] = NAN; r["B"] = INFINITY; r["C"] = -0.5; r["D"] = -INFINITY;
	w.Append(r);
	CHECK_EQ(ReadLines(p)[1], std::string("nan,inf,-0.5,-inf"));
	CHECK_EQ(MetricsCSVWriter::FormatValue(-NAN), std::string("nan"));
	std::filesystem::remove(p);
}

TEST(CSV_fehlender_Schluessel_bleibt_leeres_Feld) {
	auto p = TempCSV("missing");
	MetricsCSVWriter w(p);
	Report r1; r1["A"] = 1; r1["B"] = 2;
	w.Append(r1);
	Report r2; r2["A"] = 3;
	w.Append(r2);
	CHECK_EQ(ReadLines(p)[2], std::string("3,"));
	std::filesystem::remove(p);
}

TEST(CSV_grosse_Ganzzahlen_bleiben_exakt) {
	CHECK_EQ(MetricsCSVWriter::FormatValue(2704829056.0), std::string("2704829056"));
	CHECK_EQ(MetricsCSVWriter::FormatValue(0.0027), std::string("0.0027"));
	CHECK_EQ(MetricsCSVWriter::FormatValue(0), std::string("0"));
}

TEST(CSV_Kopfzeile_mit_Kommas_in_Anfuehrungszeichen_wird_geparst) {
	auto cols = MetricsCSVWriter::ParseHeader("\"A\",\"B, mit Komma\",\"C\"\r");
	CHECK_EQ((int)cols.size(), 3);
	CHECK_EQ(cols[1], std::string("B, mit Komma"));
	CHECK_EQ(cols[2], std::string("C"));
}

TEST(CSV_leere_Datei_gilt_als_neu) {
	auto p = TempCSV("leer");
	{ std::ofstream f(p); }   // 0 Bytes
	MetricsCSVWriter w(p);
	Report r; r["A"] = 1;
	w.Append(r);
	auto lines = ReadLines(p);
	CHECK_EQ((int)lines.size(), 2);
	CHECK_EQ(lines[0], std::string("\"A\""));
	std::filesystem::remove(p);
}
