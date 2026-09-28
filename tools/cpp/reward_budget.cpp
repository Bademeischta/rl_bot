// Reward-Bilanz je Komponente über den echten Reward-Code (Spieltest-Auftrag, Punkt 2: farmt der Bot
// Shaping statt zu schießen?).
//
//   reward_budget.exe --config train/configs/lucy_1v1_zero_sum.json [--policy <PPO_POLICY.lt>]
//                     [--games 200] [--mode sample|argmax|argmax_group] [--out budget.json]
//
// Teil 1, feste Lagen: Autos und Ball werden in einer Arena gesetzt, jede Komponente aus
// BuildLucyRewardParts wird auf diesem Zustand ausgewertet (Gewicht wie in der Config), dazu die
// Zero-Sum-Form (team_spirit > 0) wie ZeroSumReward.
// Teil 2, gespielte Lagen (mit --policy): Selbstspiel wie im Duell (300-s-Spiele, Anstoß nach Toren).
// Jede Komponente läuft als eigene Instanz neben dem Reward des Matches mit (gleiche Aufrufe: Reset,
// PreStep, GetAllRewards); die Summe muss den Reward des Matches treffen ("max_abs_diff"). Die
// Schritte werden nach Lage eingeteilt (Ball in der Ecke des Angriffsdrittels, vor dem Tor, im
// Mittelfeld) und je Lage aus Sicht des angreifenden Teams gemittelt.
//
// Dazu je Lage der Vergleich mit dem Torwert: Ein Tor ist nach dem Zero-Sum-Wrapper goal + concede
// wert und beendet die Episode. "goal_equivalent_s" = so viele Sekunden dieser Lage bringen so viel
// Shaping wie ein Tor; "hold_value_Ns" = diskontierter Wert (gae_gamma), die Lage N Sekunden zu halten.
#include "policy_io.h"

#include "env/cpp/ActionSelect.h"
#include "env/cpp/Obs.h"
#include "env/cpp/PlayStats.h"
#include "env/cpp/Rewards.h"
#include "env/cpp/StateSetters.h"
#include "train/cpp/Config.h"

#include <RLGymSim_CPP/Gym.h>
#include <RLGymSim_CPP/Math.h>
#include <RLGymSim_CPP/Utils/ActionParsers/DiscreteAction.h>
#include <RLGymSim_CPP/Utils/TerminalConditions/GoalScoreCondition.h>

#include <ATen/CPUGeneratorImpl.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

using namespace RLGSC;
using namespace RLbot;
using nlohmann::json;

namespace {

constexpr float CORNER_X = 1800.f;   // Ecke: Ball im Angriffsdrittel und |x| darüber

// Zero-Sum-Form einer Komponente je Spieler (wie RLGSC::ZeroSumReward, opponentScale 1)
std::vector<double> ZeroSum(const std::vector<double>& v, const GameState& s, float tau) {
	if (tau <= 0)
		return v;
	double sum[2] = { 0, 0 };
	int count[2] = { 0, 0 };
	for (size_t i = 0; i < v.size(); i++) {
		int t = PlayTracker::TeamOf(s.players[i]);
		sum[t] += v[i];
		count[t]++;
	}
	std::vector<double> out(v.size());
	for (size_t i = 0; i < v.size(); i++) {
		int t = PlayTracker::TeamOf(s.players[i]);
		double own = sum[t] / std::max(count[t], 1), opp = sum[1 - t] / std::max(count[1 - t], 1);
		out[i] = v[i] * (1 - tau) + own * tau - opp;
	}
	return out;
}

struct Components {
	std::vector<RewardPart> parts;
	float tau;
	explicit Components(const RewardWeights& w) : parts(BuildLucyRewardParts(w)), tau(w.teamSpirit) {}
	~Components() { for (auto& p : parts) delete p.fn; }
	void Reset(const GameState& s) { for (auto& p : parts) p.fn->Reset(s); }
	// Roh (gewichtet) und Zero-Sum je Komponente und Spieler, gleiche Aufrufe wie Match::GetRewards
	void Eval(const GameState& s, const ActionSet& prev, bool done,
	          std::vector<std::vector<double>>& raw, std::vector<std::vector<double>>& zs) {
		raw.assign(parts.size(), {});
		zs.assign(parts.size(), {});
		for (size_t c = 0; c < parts.size(); c++) {
			parts[c].fn->PreStep(s);
			auto r = parts[c].fn->GetAllRewards(s, prev, done);
			raw[c].resize(r.size());
			for (size_t i = 0; i < r.size(); i++)
				raw[c][i] = r[i] * parts[c].weight;
			zs[c] = ZeroSum(raw[c], s, tau);
		}
	}
};

class FuncSetter : public StateSetter {
public:
	std::function<void(Arena*)> fn;
	explicit FuncSetter(std::function<void(Arena*)> fn) : fn(std::move(fn)) {}
	virtual GameState ResetState(Arena* arena) { fn(arena); return GameState(arena); }
};

struct Scene {
	std::string name;
	Vec ballPos, ballVel;
	Vec aPos, aVel;          // Blau greift das orange Tor (+y) an
	Vec dPos, dVel;          // Orange
};

std::vector<Scene> StaticScenes() {
	return {
		{ "ecke_verteidiger_im_tor", Vec(3250, 4550, 93), Vec(-300, 100, 0),
		  Vec(3450, 4450, 17), Vec(-350, 120, 0), Vec(0, 5050, 17), Vec(0, 0, 0) },
		{ "ecke_verteidiger_greift_an", Vec(3250, 4550, 93), Vec(-300, 100, 0),
		  Vec(3450, 4450, 17), Vec(-350, 120, 0), Vec(1200, 4900, 17), Vec(900, -150, 0) },
		{ "grundlinie_quer", Vec(-400, 4700, 93), Vec(900, 0, 0),
		  Vec(-600, 4700, 17), Vec(900, 0, 0), Vec(0, 5050, 17), Vec(0, 0, 0) },
		{ "dribbling_mittelfeld", Vec(0, 0, 160), Vec(0, 1200, 0),
		  Vec(0, -50, 17), Vec(0, 1200, 0), Vec(0, 3500, 17), Vec(0, 1000, 0) },
		{ "schussposition_frontal", Vec(0, 3500, 93), Vec(0, 600, 0),
		  Vec(0, 3300, 17), Vec(0, 1400, 0), Vec(0, 5050, 17), Vec(0, 0, 0) },
		{ "neutral_beide_zum_ball", Vec(0, 0, 93), Vec(0, 0, 0),
		  Vec(0, -1500, 17), Vec(0, 1400, 0), Vec(0, 1500, 17), Vec(0, -1400, 0) },
	};
}

// Kennzahlen einer Lage aus Sicht des Angreifers (Blau in den festen Lagen)
json LageJSON(const std::vector<std::string>& names, const std::vector<double>& rawA, const std::vector<double>& rawD,
              const std::vector<double>& zsA, double goalValue, float gamma, int tickSkip) {
	double stepsPerSec = 120.0 / tickSkip;
	json comps = json::object();
	double sumRawA = 0, sumRawD = 0, sumZs = 0;
	for (size_t c = 0; c < names.size(); c++) {
		comps[names[c]] = { { "attacker_raw", rawA[c] }, { "defender_raw", rawD[c] },
		                    { "zero_sum", zsA[c] }, { "zero_sum_per_s", zsA[c] * stepsPerSec } };
		sumRawA += rawA[c];
		sumRawD += rawD[c];
		sumZs += zsA[c];
	}
	json j = { { "components", comps }, { "attacker_raw", sumRawA }, { "defender_raw", sumRawD },
	           { "zero_sum", sumZs }, { "zero_sum_per_s", sumZs * stepsPerSec }, { "goal_value", goalValue } };
	if (sumZs > 0) {
		j["goal_equivalent_s"] = goalValue / (sumZs * stepsPerSec);
		for (int secs : { 1, 3, 5 }) {
			double n = secs * stepsPerSec;
			j["hold_value_" + std::to_string(secs) + "s"] = sumZs * (1 - std::pow((double)gamma, n)) / (1 - gamma);
		}
	}
	return j;
}

struct Accum {
	std::vector<double> rawA, rawD, zsA;
	int64_t steps = 0;
	void Add(const std::vector<std::vector<double>>& raw, const std::vector<std::vector<double>>& zs,
	         const std::vector<int>& attackers, const std::vector<int>& defenders) {
		size_t n = raw.size();
		if (rawA.empty()) { rawA.assign(n, 0); rawD.assign(n, 0); zsA.assign(n, 0); }
		for (size_t c = 0; c < n; c++) {
			double a = 0, d = 0, z = 0;
			for (int i : attackers) { a += raw[c][i]; z += zs[c][i]; }
			for (int i : defenders) d += raw[c][i];
			rawA[c] += a / attackers.size();
			rawD[c] += d / std::max<size_t>(defenders.size(), 1);
			zsA[c] += z / attackers.size();
		}
		steps++;
	}
	void Merge(const Accum& o) {
		if (o.steps == 0) return;
		if (rawA.empty()) { *this = o; return; }
		for (size_t c = 0; c < rawA.size(); c++) { rawA[c] += o.rawA[c]; rawD[c] += o.rawD[c]; zsA[c] += o.zsA[c]; }
		steps += o.steps;
	}
	std::vector<double> Mean(const std::vector<double>& v) const {
		std::vector<double> m(v.size());
		for (size_t i = 0; i < v.size(); i++) m[i] = steps ? v[i] / steps : 0;
		return m;
	}
};

} // namespace

int main(int argc, char** argv) {
	std::string configPath, policyPath, outPath = "reward_budget.json", meshDir = "collision_meshes", modeName = "sample";
	int games = 200, seconds = 300, threads = 0, seed = 123;
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		auto next = [&]() -> std::string {
			if (i + 1 >= argc) { std::cerr << "Fehlender Wert für " << arg << "\n"; exit(2); }
			return argv[++i];
		};
		if (arg == "--config") configPath = next();
		else if (arg == "--policy") policyPath = next();
		else if (arg == "--out") outPath = next();
		else if (arg == "--meshes") meshDir = next();
		else if (arg == "--games") games = std::stoi(next());
		else if (arg == "--seconds") seconds = std::stoi(next());
		else if (arg == "--threads") threads = std::stoi(next());
		else if (arg == "--seed") seed = std::stoi(next());
		else if (arg == "--mode") modeName = next();
		else { std::cerr << "Unbekanntes Argument: " << arg << "\n"; return 2; }
	}
	SelectMode mode;
	if (configPath.empty() || !ParseSelectMode(modeName, mode)) {
		std::cerr << "usage: reward_budget --config <cfg.json> [--policy <PPO_POLICY.lt>] [--games N] "
		             "[--mode sample|argmax|argmax_group] [--out budget.json]\n";
		return 2;
	}
	TrainConfig cfg = TrainConfig::FromFile(configPath);
	RocketSim::Init(meshDir);
	const float tau = cfg.rewards.teamSpirit;
	const double goalValue = tau > 0 ? cfg.rewards.goal + cfg.rewards.concede : cfg.rewards.goal;

	std::vector<std::string> names;
	{
		Components probe(cfg.rewards);
		for (auto& p : probe.parts) names.push_back(p.name);
	}

	json out = { { "config", configPath }, { "team_spirit", tau }, { "gae_gamma", cfg.gaeGamma },
	             { "tick_skip", cfg.tickSkip }, { "goal_value", goalValue }, { "components", names } };

	// --- Teil 1: feste Lagen ---
	json scenes = json::object();
	for (auto& sc : StaticScenes()) {
		auto* match = new Match(BuildLucyReward(cfg.rewards), { new GoalScoreCondition() }, new StackedPaddedOBS(3, 5, false),
		                        new DiscreteAction(), new FuncSetter([sc](Arena* a) {
			a->ResetToRandomKickoff(0);
			BallState bs = {};
			bs.pos = sc.ballPos;
			bs.vel = sc.ballVel;
			a->ball->SetState(bs);
			for (Car* car : a->_cars) {
				bool blue = car->team == Team::BLUE;
				CarState cs = {};
				cs.pos = blue ? sc.aPos : sc.dPos;
				cs.vel = blue ? sc.aVel : sc.dVel;
				Vec look = sc.ballPos - cs.pos;
				cs.rotMat = Angle(std::atan2(look.y, look.x), 0, 0).ToRotMat();
				cs.boost = 30;
				car->SetState(cs);
			}
		}), 1, true);
		Gym gym(match, cfg.tickSkip);
		gym.Reset();
		const GameState& s = gym.prevState;
		Components comps(cfg.rewards);
		comps.Reset(s);
		std::vector<std::vector<double>> raw, zs;
		comps.Eval(s, ActionSet(s.players.size()), false, raw, zs);
		int a = s.players[0].team == Team::BLUE ? 0 : 1, d = 1 - a;
		std::vector<double> rawA, rawD, zsA;
		for (size_t c = 0; c < names.size(); c++) {
			rawA.push_back(raw[c][a]);
			rawD.push_back(raw[c][d]);
			zsA.push_back(zs[c][a]);
		}
		scenes[sc.name] = LageJSON(names, rawA, rawD, zsA, goalValue, cfg.gaeGamma, cfg.tickSkip);
		delete match;
	}
	out["static"] = scenes;

	// --- Teil 2: gespielte Lagen ---
	if (!policyPath.empty()) {
		torch::NoGradGuard noGrad;
		torch::set_num_threads(1);
		LoadedPolicy policy = LoadPolicy(policyPath);
		std::map<std::string, Accum> total;
		double maxDiff = 0;
		int64_t goals = 0;
		std::mutex mutex;
		std::atomic<int> nextGame = 0;
		int nThreads = threads > 0 ? threads : (int)std::max(1u, std::thread::hardware_concurrency() / 2);
		auto worker = [&]() {
			torch::NoGradGuard g;
			torch::set_num_threads(1);
			auto seq = torch::nn::Sequential(std::dynamic_pointer_cast<torch::nn::SequentialImpl>(policy.seq->clone()));
			std::map<std::string, Accum> local;
			double localDiff = 0;
			int64_t localGoals = 0;
			const auto groundGroups = EffectGroups(DiscreteAction().actions, true);
			const auto airGroups = EffectGroups(DiscreteAction().actions, false);
			for (int game = nextGame++; game < games; game = nextGame++) {
				auto* match = new Match(BuildLucyReward(cfg.rewards), { new GoalScoreCondition() },
				                        new StackedPaddedOBS(3, 5, false), new DiscreteAction(), new KickoffSetter(), 1, true);
				Gym gym(match, cfg.tickSkip);
				at::Generator gen = at::detail::createCPUGenerator((uint64_t)seed * 7919ull + game);
				Components comps(cfg.rewards);
				FList2 obs = gym.Reset();
				comps.Reset(gym.prevState);
				GameState state = gym.prevState;
				std::vector<std::vector<double>> raw, zs;
				for (int step = 0; step < seconds * 120 / cfg.tickSkip; step++) {
					IList actions(state.players.size());
					for (size_t p = 0; p < state.players.size(); p++) {
						auto input = torch::from_blob(obs[p].data(), { 1, (int64_t)obs[p].size() }).clone();
						auto probs = PolicyProbs(seq, input);
						if (mode == SelectMode::ARGMAX) actions[p] = (int)probs.argmax(1).item<int64_t>();
						else if (mode == SelectMode::ARGMAX_GROUP)
							actions[p] = ArgmaxGroup(probs.contiguous().data_ptr<float>(),
							                         state.players[p].carState.isOnGround ? groundGroups : airGroups);
						else actions[p] = (int)torch::multinomial(probs, 1, true, gen).item<int64_t>();
					}
					auto r = gym.Step(actions);
					comps.Eval(r.state, match->prevActions, r.done, raw, zs);
					for (size_t p = 0; p < r.reward.size(); p++) {
						double sum = 0;
						for (size_t c = 0; c < names.size(); c++) sum += zs[c][p];
						localDiff = std::max(localDiff, std::abs(sum - r.reward[p]));
					}
					// Lage aus Sicht des Teams, in dessen Angriffsdrittel der Ball ist
					int att = PlayTracker::AttackingThird(r.state.ball.pos.y);
					std::vector<int> attackers, defenders;
					for (size_t p = 0; p < r.state.players.size(); p++)
						(PlayTracker::TeamOf(r.state.players[p]) == (att >= 0 ? att : 0) ? attackers : defenders).push_back((int)p);
					std::string lage = att < 0 ? "mittelfeld_blau_sicht"
						: (std::abs(r.state.ball.pos.x) > CORNER_X ? "angriffsdrittel_ecke" : "angriffsdrittel_mitte");
					if (!r.done) {   // Torschritt gesondert (enthält den Torwert)
						local[lage].Add(raw, zs, attackers, defenders);
						if (att >= 0) local["angriffsdrittel_alle"].Add(raw, zs, attackers, defenders);
					}
					if (r.done) {
						localGoals++;
						obs = gym.Reset();
						comps.Reset(gym.prevState);
						state = gym.prevState;
					} else {
						obs = r.obs;
						state = r.state;
					}
				}
				delete match;
			}
			std::lock_guard<std::mutex> lock(mutex);
			for (auto& [k, v] : local) total[k].Merge(v);
			maxDiff = std::max(maxDiff, localDiff);
			goals += localGoals;
		};
		std::vector<std::thread> pool;
		for (int t = 0; t < std::min(nThreads, games); t++) pool.emplace_back(worker);
		for (auto& t : pool) t.join();

		json played = json::object();
		int64_t allSteps = 0;
		for (auto& [k, v] : total) allSteps += k == "angriffsdrittel_alle" ? 0 : v.steps;
		for (auto& [k, v] : total) {
			json j = LageJSON(names, v.Mean(v.rawA), v.Mean(v.rawD), v.Mean(v.zsA), goalValue, cfg.gaeGamma, cfg.tickSkip);
			j["steps"] = v.steps;
			j["time_share"] = allSteps ? (double)v.steps / allSteps : 0.0;
			played[k] = j;
		}
		out["played"] = played;
		out["policy"] = policyPath;
		out["mode"] = modeName;
		out["games"] = games;
		out["goals"] = goals;
		out["max_abs_diff"] = maxDiff;
		std::cout << "Gespielt: " << games << " Spiele, " << goals << " Tore, Abweichung Komponentensumme zu "
		          << "Match-Reward max " << maxDiff << "\n";
	}

	std::ofstream(outPath) << out.dump(2);
	std::cout << "Ergebnis in " << outPath << "\n";
	return 0;
}
