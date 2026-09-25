#include "Metrics.h"

#include "../../env/cpp/StateSetters.h"
#include "../../env/cpp/TimeoutCondition.h"

#include <RLGymSim_CPP/Math.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/NoTouchCondition.h>

#include <map>

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
		}
	}
	return end;
}

void AccumEpisodeEnd(const EpisodeEnd& end, uint64_t episodeSteps, const std::string& sceneName,
                     Report& metrics) {
	metrics.AccumAvg("ep_end_goal", end.goal ? 1 : 0);
	metrics.AccumAvg("ep_end_timeout", end.goal ? 0 : 1);
	metrics.AccumAvg("ep_end_notouch", (!end.goal && end.noTouch) ? 1 : 0);
	metrics.AccumAvg("ep_end_time", (!end.goal && !end.noTouch && end.timeLimit) ? 1 : 0);
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

} // namespace RLbot
