// Eigene Trainingsmetriken (Audit M8): Episoden-Endgrund, Episodenlänge, Szenen-Aufschlüsselung.
//
// Die Funktionen sind bewusst frei von Learner-Zustand, damit sie in tests/cpp/test_metrics.cpp
// ohne laufendes Training geprüft werden können. main.cpp verdrahtet sie in die Callbacks.
//
// Schlüssel, die pro Iteration in metrics.csv landen (alle als Mittelwert über die Iteration):
//   player_speed, ball_touch_ratio, in_air_ratio, boost_held, supersonic_ratio,
//   ball_speed, ball_height                              (wie bisher, pro Step)
//   ep_end_goal      Anteil der Episodenenden mit Tor
//   ep_end_timeout   Anteil ohne Tor (NoTouch- oder Spielzeit-Timeout)
//   ep_end_notouch   Anteil NoTouch-Timeout
//   ep_end_time      Anteil Spielzeit-Timeout (game_timeout_secs)
//   ep_end_truncated Anteil, den der Upstream als Truncation meldet (nur mit K1-Patch)
//   ep_length_steps  mittlere Episodenlänge in Steps (Steps * tick_skip / 120 = Sekunden)
//   scene_<name>_goal, scene_<name>_length   dasselbe je State-Setter-Szene
//   raw_step_reward  Reward pro Spieler und Step VOR einem Zero-Sum-Wrapper (Review R10); ohne
//                    Wrapper gleich "Average Step Reward", mit Zero-Sum im 1v1 ist der sonst 0
#pragma once

#include <RLGymPPO_CPP/Threading/GameInst.h>

#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace RLbot {

// Pro-Step-Metriken über alle Spieler und den Ball.
void AccumStepMetrics(const RLGSC::GameState& state, RLGPC::Report& metrics);

// raw_step_reward aus dem RawRewardTap des Matches (nichts, wenn es keinen gibt).
void AccumRawReward(const RLGSC::Match* match, RLGPC::Report& metrics);

// Wie eine Episode geendet hat, abgeleitet aus dem Zustand und den Terminal-Bedingungen.
struct EpisodeEnd {
	bool goal = false;       // Ball im Tor
	bool noTouch = false;    // NoTouchCondition hat ausgelöst
	bool timeLimit = false;  // TimeoutCondition (game_timeout_secs) hat ausgelöst
	bool truncated = false;  // vom Upstream gemeldet (K1-Patch), sonst false
};
EpisodeEnd ClassifyEpisodeEnd(const RLGSC::GameState& state, const RLGSC::Match* match, bool truncatedFlag);

// Episoden-Ende-Metriken eintragen. sceneName darf leer sein (dann keine Szenen-Aufschlüsselung).
void AccumEpisodeEnd(const EpisodeEnd& end, uint64_t episodeSteps, const std::string& sceneName,
                     RLGPC::Report& metrics);

// Name der Szene, die die laufende Episode gestartet hat (leer, wenn kein WeightedStateSetter).
std::string CurrentSceneName(const RLGSC::Match* match);

// Zählt die Steps seit dem letzten Episodenende je GameInst. Wird aus vielen Threads
// aufgerufen, sperrt aber nur am Episodenende (rund 40-mal pro Iteration, nicht pro Step).
class EpisodeLengthTracker {
public:
	// Aufruf am Ende einer Episode; stepsDone = Zahl der abgeschlossenen Steps des Spiels
	// inklusive des aktuellen. Liefert die Länge der gerade beendeten Episode.
	uint64_t OnEpisodeEnd(const void* gameKey, uint64_t stepsDone);

private:
	std::mutex mutex;
	std::unordered_map<const void*, uint64_t> lastEnd;
};

// Mittelt alle mit Report::AccumAvg gesammelten Schlüssel über die Spiel-Reports einer Iteration.
// Anders als eine feste Schlüsselliste nimmt das auch Schlüssel mit, die erst später auftauchen
// (Szenen, Skill-Ratings je Modus), siehe Audit M1.
void AggregateGameMetrics(const std::vector<RLGPC::Report>& gameReports, RLGPC::Report& out);

// Schreibt pro Iteration eine Zeile nach metrics.csv (Audit M1, N4).
//  - Kopfzeile nur, wenn die Datei neu ist; beim Fortsetzen wird die vorhandene Kopfzeile
//    übernommen (keine wiederholten Kopfzeilen mehr mitten in der Datei)
//  - taucht ein neuer Schlüssel auf (z. B. "Skill Rating 2v2" nach der ersten Eval), wird er
//    hinten angehängt und die Kopfzeile in Zeile 1 einmal neu geschrieben; ältere Zeilen
//    bleiben kürzer, was CSV-Leser als leere Felder lesen
//  - nicht-endliche Werte werden wörtlich als nan, inf, -inf geschrieben (Review-Befund R5:
//    vorher als leeres Feld, das tools/experiments/check_abort.py nicht als Abbruch erkannte).
//    Ein leeres Feld heißt nur noch: Schlüssel fehlte in dieser Iteration.
//  - Zahlen mit 12 signifikanten Stellen, damit "Cumulative Timesteps" exakt bleibt
class MetricsCSVWriter {
public:
	explicit MetricsCSVWriter(std::filesystem::path path = {}) : path(std::move(path)) {}

	void Append(const RLGPC::Report& report);

	const std::vector<std::string>& Columns() const { return columns; }
	static std::vector<std::string> ParseHeader(const std::string& line);
	static std::string FormatValue(double value);

private:
	std::filesystem::path path;
	std::vector<std::string> columns;
	bool initialized = false;

	// Liest die Kopfzeile einer vorhandenen Datei; liefert false, wenn die Datei fehlt oder leer ist.
	bool LoadExistingHeader();
	void RewriteHeader();
};

} // namespace RLbot
