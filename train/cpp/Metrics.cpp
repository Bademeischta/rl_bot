#include "Metrics.h"

#include "../../env/cpp/Rewards.h"
#include "../../env/cpp/StateSetters.h"
#include "../../env/cpp/TimeoutCondition.h"

#include <RLGymSim_CPP/Math.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/NoTouchCondition.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>

namespace RLbot {

using RLGPC::Report;
using RLGSC::GameState;
using RLGSC::Match;

void AccumStepMetrics(const GameState& state, Report& metrics) {
	for (auto& player : state.players) {
		metrics.AccumAvg("player_speed", player.phys.vel.Length());
		metrics.AccumAvg("ball_touch_ratio", player.ballTouchedStep);
		metrics.AccumAvg("in_air_ratio", !player.carState.isOnGround);
		metrics.AccumAvg("boost_held", player.boostFraction);
		metrics.AccumAvg("supersonic_ratio", player.carState.isSupersonic);
	}
	metrics.AccumAvg("ball_speed", state.ball.vel.Length());
	metrics.AccumAvg("ball_height", state.ball.pos.z);
}

void AccumRawReward(const Match* match, Report& metrics) {
	if (!match)
		return;
	const RawRewardTap* tap = FindRawRewardTap(match->rewardFn);
	if (!tap || tap->lastRaw.empty())
		return;
	double sum = 0;
	for (float r : tap->lastRaw)
		sum += r;
	metrics.AccumAvg("raw_step_reward", sum / tap->lastRaw.size());
}

EpisodeEnd ClassifyEpisodeEnd(const GameState& state, const Match* match, bool truncatedFlag) {
	EpisodeEnd end;
	end.goal = RLGSC::Math::IsBallScored(state.ball.pos);
	end.truncated = truncatedFlag;
	if (match) {
		for (auto* cond : match->terminalConditions) {
			if (auto* noTouch = dynamic_cast<RLGSC::NoTouchCondition*>(cond))
				end.noTouch |= noTouch->stepsSinceTouch >= noTouch->maxSteps;
			else if (auto* timeout = dynamic_cast<TimeoutCondition*>(cond))
				end.timeLimit |= timeout->steps >= timeout->maxSteps;
			else if (auto* drill = dynamic_cast<SceneTimeoutCondition*>(cond))
				end.drill |= drill->Fired();
		}
	}
	return end;
}

void AccumEpisodeEnd(const EpisodeEnd& end, uint64_t episodeSteps, const std::string& sceneName,
                     Report& metrics) {
	// Ein Anstoß-Drill endet planmäßig: kein Timeout im Sinn von K1 (ep_end_timeout zählt nur Zeitlimits)
	bool drill = !end.goal && end.drill && !end.noTouch && !end.timeLimit;
	metrics.AccumAvg("ep_end_goal", end.goal ? 1 : 0);
	metrics.AccumAvg("ep_end_timeout", (end.goal || drill) ? 0 : 1);
	metrics.AccumAvg("ep_end_notouch", (!end.goal && end.noTouch) ? 1 : 0);
	metrics.AccumAvg("ep_end_time", (!end.goal && !end.noTouch && end.timeLimit) ? 1 : 0);
	metrics.AccumAvg("ep_end_drill", drill ? 1 : 0);
	metrics.AccumAvg("ep_end_truncated", end.truncated ? 1 : 0);
	metrics.AccumAvg("ep_length_steps", (double)episodeSteps);
	if (!sceneName.empty()) {
		metrics.AccumAvg("scene_" + sceneName + "_goal", end.goal ? 1 : 0);
		metrics.AccumAvg("scene_" + sceneName + "_length", (double)episodeSteps);
	}
}

std::string CurrentSceneName(const Match* match) {
	if (!match)
		return {};
	auto* setter = dynamic_cast<WeightedStateSetter*>(match->stateSetter);
	if (!setter || setter->lastPicked < 0 || setter->lastPicked >= (int)setter->names.size())
		return {};
	return setter->names[setter->lastPicked];
}

namespace {

// Schlüsselnamen je Präfix einmal bauen (der Callback läuft pro Step in 16 Threads)
struct PlayKeys {
	std::string goalsPerMin, shotsPerMin, airPerMin, aerialPerMin, touchHeight, airHeight, airShare,
		koTime, koSpeed, koLoserSpeed, koBoost, koGoal, koUntouched,
		thirdShare, spell, conversion, noGoal, longSpell, mateDist, doubleCommit, lastBack;

	explicit PlayKeys(const std::string& p)
		: goalsPerMin(p + "goals_per_min"), shotsPerMin(p + "shots_per_min"),
		  airPerMin(p + "air_touch_per_min"), aerialPerMin(p + "aerial_touch_per_min"),
		  touchHeight(p + "touch_height_mean"), airHeight(p + "air_touch_height_mean"),
		  airShare(p + "air_touch_share"),
		  koTime(p + "kickoff_first_touch_s"), koSpeed(p + "kickoff_touch_speed"),
		  koLoserSpeed(p + "kickoff_loser_speed"), koBoost(p + "kickoff_boost_used"),
		  koGoal(p + "kickoff_goal_10s"), koUntouched(p + "kickoff_untouched"),
		  thirdShare(p + "off_third_share"), spell(p + "off_third_spell_s"),
		  conversion(p + "off_third_conversion"), noGoal(p + "off_third_nogoal_s"),
		  longSpell(p + "off_third_long"),
		  mateDist(p + "mate_dist"), doubleCommit(p + "double_commit"), lastBack(p + "last_back") {}
};

const PlayKeys& KeysFor(int teamSize) {
	static const PlayKeys k1(""), k2("2v2_"), k3("3v3_");
	return teamSize >= 3 ? k3 : (teamSize == 2 ? k2 : k1);
}

} // namespace

void AccumPlayMetrics(const PlayEvents& ev, int teamSize, int tickSkip, Report& m) {
	const PlayKeys& k = KeysFor(teamSize);
	const double stepsPerMin = 60.0 * 120.0 / tickSkip;
	const double players = ev.players > 0 ? ev.players : 1;

	int air = 0, aerial = 0;
	for (auto& t : ev.touches) {
		m.AccumAvg(k.touchHeight, t.ballHeight);
		m.AccumAvg(k.airShare, t.carInAir ? 1 : 0);
		if (t.carInAir) {
			air++;
			m.AccumAvg(k.airHeight, t.ballHeight);
			if (t.ballHeight >= AERIAL_TOUCH_MIN_HEIGHT)
				aerial++;
		}
	}
	m.AccumAvg(k.goalsPerMin, ev.goalTeam >= 0 ? stepsPerMin : 0.0);
	m.AccumAvg(k.shotsPerMin, (ev.shots[0] + ev.shots[1]) * stepsPerMin / players);
	m.AccumAvg(k.airPerMin, air * stepsPerMin / players);
	m.AccumAvg(k.aerialPerMin, aerial * stepsPerMin / players);
	m.AccumAvg(k.thirdShare, ev.ballThirdTeam >= 0 ? 1 : 0);

	for (auto& ko : ev.kickoffs) {
		m.AccumAvg(k.koUntouched, ko.firstTeam < 0 ? 1 : 0);
		m.AccumAvg(k.koGoal, ko.goalTeam >= 0 ? 1 : 0);
		m.AccumAvg(k.koBoost, (ko.boostUsed[0] + ko.boostUsed[1]) / 2.0);
		if (ko.firstTeam >= 0) {
			m.AccumAvg(k.koTime, ko.timeToTouch);
			m.AccumAvg(k.koSpeed, ko.touchSpeed);
			m.AccumAvg(k.koLoserSpeed, ko.loserSpeed);
		}
	}
	for (auto& s : ev.spells) {
		m.AccumAvg(k.spell, s.seconds);
		m.AccumAvg(k.conversion, s.goal ? 1 : 0);
		m.AccumAvg(k.longSpell, s.seconds >= OFF_THIRD_LONG_SECS ? 1 : 0);
		if (!s.goal)
			m.AccumAvg(k.noGoal, s.seconds);
	}
	for (auto& t : ev.teams) {
		m.AccumAvg(k.mateDist, t.mateDist);
		m.AccumAvg(k.doubleCommit, t.doubleCommit ? 1 : 0);
		m.AccumAvg(k.lastBack, t.lastBack ? 1 : 0);
	}
}

PlayTracker& PlayTrackerFor(const void* gameKey, int tickSkip) {
	thread_local std::unordered_map<const void*, std::unique_ptr<PlayTracker>> trackers;
	auto& tracker = trackers[gameKey];
	if (!tracker)
		tracker = std::make_unique<PlayTracker>(tickSkip);
	return *tracker;
}

uint64_t EpisodeLengthTracker::OnEpisodeEnd(const void* gameKey, uint64_t stepsDone) {
	std::lock_guard<std::mutex> lock(mutex);
	uint64_t& last = lastEnd[gameKey];
	uint64_t length = stepsDone - last;
	last = stepsDone;
	return length;
}

void AggregateGameMetrics(const std::vector<Report>& gameReports, Report& out) {
	static const std::string TOTAL = "_avg_total", COUNT = "_avg_count";
	std::map<std::string, std::pair<double, double>> sums; // name -> (total, count)

	for (auto& report : gameReports) {
		for (auto& pair : report.data) {
			const std::string& key = pair.first;
			if (key.size() <= TOTAL.size() || key.compare(key.size() - TOTAL.size(), TOTAL.size(), TOTAL) != 0)
				continue;
			std::string name = key.substr(0, key.size() - TOTAL.size());
			auto countIt = report.data.find(name + COUNT);
			if (countIt == report.data.end())
				continue;
			auto& entry = sums[name];
			entry.first += pair.second;
			entry.second += countIt->second;
		}
	}

	for (auto& pair : sums)
		if (pair.second.second > 0)
			out[pair.first] = pair.second.first / pair.second.second;
}

// --- metrics.csv -----------------------------------------------------------

static bool IsInternalAvgKey(const std::string& key) {
	return key.find("_avg_total") != std::string::npos || key.find("_avg_count") != std::string::npos;
}

std::vector<std::string> MetricsCSVWriter::ParseHeader(const std::string& line) {
	std::vector<std::string> out;
	std::string cur;
	bool inQuotes = false;
	for (size_t i = 0; i < line.size(); i++) {
		char c = line[i];
		if (c == '"') {
			inQuotes = !inQuotes;
		} else if (c == ',' && !inQuotes) {
			out.push_back(cur);
			cur.clear();
		} else if (c != '\r') {
			cur += c;
		}
	}
	out.push_back(cur);
	return out;
}

std::string MetricsCSVWriter::FormatValue(double value) {
	// Wörtlich statt leer (Review R5), und plattformunabhängig (MSVC schreibt sonst "-nan(ind)")
	if (std::isnan(value))
		return "nan";
	if (std::isinf(value))
		return value > 0 ? "inf" : "-inf";
	std::ostringstream s;
	s << std::setprecision(12) << value;
	return s.str();
}

bool MetricsCSVWriter::LoadExistingHeader() {
	std::error_code ec;
	if (!std::filesystem::exists(path, ec) || std::filesystem::file_size(path, ec) == 0)
		return false;
	std::ifstream fIn(path);
	std::string line;
	if (!std::getline(fIn, line) || line.empty())
		return false;
	columns = ParseHeader(line);
	return true;
}

void MetricsCSVWriter::RewriteHeader() {
	// Nur Zeile 1 ändert sich; die Datei wird einmal komplett umkopiert (passiert nur, wenn
	// ein neuer Schlüssel auftaucht, also selten).
	std::ifstream fIn(path);
	if (!fIn.good())
		return;
	std::string firstLine;
	std::getline(fIn, firstLine);
	std::filesystem::path tmp = path;
	tmp += ".tmp";
	{
		std::ofstream fOut(tmp, std::ios::binary);
		for (size_t i = 0; i < columns.size(); i++)
			fOut << (i ? "," : "") << '"' << columns[i] << '"';
		fOut << '\n';
		fOut << fIn.rdbuf();
	}
	fIn.close();
	std::error_code ec;
	std::filesystem::rename(tmp, path, ec);
	if (ec) {
		// Fallback (z. B. Windows mit offenem Handle): Inhalt kopieren statt umbenennen
		std::filesystem::copy_file(tmp, path, std::filesystem::copy_options::overwrite_existing, ec);
		std::filesystem::remove(tmp, ec);
	}
}

void MetricsCSVWriter::Append(const RLGPC::Report& report) {
	if (path.empty())
		return;

	bool fileHasData = false;
	if (!initialized) {
		fileHasData = LoadExistingHeader();
		initialized = true;
	} else {
		fileHasData = true;
	}

	bool changed = false;
	for (auto& pair : report.data) {
		if (IsInternalAvgKey(pair.first))
			continue;
		if (std::find(columns.begin(), columns.end(), pair.first) == columns.end()) {
			columns.push_back(pair.first);
			changed = true;
		}
	}

	if (fileHasData) {
		if (changed)
			RewriteHeader();
	} else {
		std::ofstream fOut(path, std::ios::binary);
		for (size_t i = 0; i < columns.size(); i++)
			fOut << (i ? "," : "") << '"' << columns[i] << '"';
		fOut << '\n';
	}

	std::ofstream fOut(path, std::ios::app | std::ios::binary);
	if (!fOut.good())
		return;
	for (size_t i = 0; i < columns.size(); i++) {
		auto it = report.data.find(columns[i]);
		fOut << (i ? "," : "");
		if (it != report.data.end())
			fOut << FormatValue(it->second);
	}
	fOut << '\n';
}

} // namespace RLbot
