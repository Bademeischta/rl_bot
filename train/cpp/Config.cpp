#include "Config.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <set>

namespace RLbot {

using nlohmann::json;

// Liest ein Feld, wenn vorhanden, und merkt sich den Namen als "gesehen".
#define READ(obj, seen, target, name) \
	do { seen.insert(name); if ((obj).contains(name)) target = (obj).at(name).get<decltype(target)>(); } while (0)

static void CheckUnknown(const json& obj, const std::set<std::string>& known, const std::string& where) {
	for (auto& item : obj.items())
		if (!known.count(item.key()))
			RG_ERR_CLOSE("Unbekanntes Config-Feld \"" << item.key() << "\" in " << where);
}

TrainConfig TrainConfig::FromFile(const std::string& path) {
	std::ifstream fIn(path);
	if (!fIn.good())
		RG_ERR_CLOSE("Config nicht lesbar: " << path);

	json j;
	try {
		j = json::parse(fIn, nullptr, true, true); // Kommentare erlaubt
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Config ist kein gültiges JSON (" << path << "): " << e.what());
	}

	TrainConfig cfg = {};
	std::set<std::string> top;

	{
		std::set<std::string> seen;
		top.insert("env");
		if (j.contains("env")) {
			auto& e = j.at("env");
			READ(e, seen, cfg.tickSkip, "tick_skip");
			READ(e, seen, cfg.maxPlayers, "max_players");
			READ(e, seen, cfg.actionStackSize, "action_stack_size");
			READ(e, seen, cfg.noTouchTimeoutSecs, "no_touch_timeout_secs");
			READ(e, seen, cfg.gameTimeoutSecs, "game_timeout_secs");
			READ(e, seen, cfg.shuffleSlots, "shuffle_slots");
			READ(e, seen, cfg.seedEnvs, "seed_envs");
			seen.insert("mode_mix");
			if (e.contains("mode_mix")) {
				auto mix = e.at("mode_mix").get<std::vector<float>>();
				if (mix.size() != 3)
					RG_ERR_CLOSE("env.mode_mix braucht genau 3 Werte (1v1, 2v2, 3v3)");
				for (int i = 0; i < 3; i++) cfg.modeMix[i] = mix[i];
			}
			CheckUnknown(e, seen, "env");
		}
	}
	{
		std::set<std::string> seen;
		top.insert("rewards");
		if (j.contains("rewards")) {
			auto& r = j.at("rewards");
			auto& w = cfg.rewards;
			READ(r, seen, w.goal, "goal");
			READ(r, seen, w.concede, "concede");
			READ(r, seen, w.demo, "demo");
			READ(r, seen, w.demoed, "demoed");
			READ(r, seen, w.touchBallToGoalAccel, "touch_ball_to_goal_accel");
			READ(r, seen, w.offensivePotential, "offensive_potential_krc");
			READ(r, seen, w.distWeightedAlign, "dist_weighted_align_krc");
			READ(r, seen, w.velocityPlayerToBall, "velocity_player_to_ball");
			READ(r, seen, w.saveBoost, "save_boost");
			READ(r, seen, w.inAir, "in_air");
			READ(r, seen, w.teamSpirit, "team_spirit");
			CheckUnknown(r, seen, "rewards");
		}
	}
	{
		std::set<std::string> seen;
		top.insert("state_setters");
		if (j.contains("state_setters")) {
			auto& s = j.at("state_setters");
			auto& w = cfg.states;
			READ(s, seen, w.kickoff, "kickoff");
			READ(s, seen, w.random, "random");
			READ(s, seen, w.aerial, "aerial");
			READ(s, seen, w.dribble, "dribble");
			READ(s, seen, w.wallPlay, "wall_play");
			READ(s, seen, w.recovery, "recovery");
			READ(s, seen, w.defense, "defense");
			CheckUnknown(s, seen, "state_setters");
		}
	}
	{
		std::set<std::string> seen;
		top.insert("learner");
		if (j.contains("learner")) {
			auto& l = j.at("learner");
			READ(l, seen, cfg.numThreads, "num_threads");
			READ(l, seen, cfg.numGamesPerThread, "num_games_per_thread");
			READ(l, seen, cfg.collectionDuringLearn, "collection_during_learn");
			READ(l, seen, cfg.device, "device");
			READ(l, seen, cfg.timestepLimit, "timestep_limit");
			READ(l, seen, cfg.extraSteps, "extra_steps");
			READ(l, seen, cfg.saveOnExit, "save_on_exit");
			READ(l, seen, cfg.timestepsPerIteration, "timesteps_per_iteration");
			READ(l, seen, cfg.timestepsPerSave, "timesteps_per_save");
			READ(l, seen, cfg.checkpointsToKeep, "checkpoints_to_keep");
			READ(l, seen, cfg.checkpointFolder, "checkpoint_folder");
			READ(l, seen, cfg.policyLayerSizes, "policy_layer_sizes");
			READ(l, seen, cfg.criticLayerSizes, "critic_layer_sizes");
			READ(l, seen, cfg.ppoEpochs, "ppo_epochs");
			READ(l, seen, cfg.expBufferIterations, "exp_buffer_iterations");
			READ(l, seen, cfg.ppoBatchSize, "ppo_batch_size");
			READ(l, seen, cfg.ppoMiniBatchSize, "ppo_mini_batch_size");
			READ(l, seen, cfg.entCoef, "ent_coef");
			READ(l, seen, cfg.clipRange, "clip_range");
			READ(l, seen, cfg.policyLR, "policy_lr");
			READ(l, seen, cfg.criticLR, "critic_lr");
			READ(l, seen, cfg.gaeLambda, "gae_lambda");
			READ(l, seen, cfg.gaeGamma, "gae_gamma");
			READ(l, seen, cfg.randomSeed, "random_seed");
			CheckUnknown(l, seen, "learner");
		}
	}
	{
		std::set<std::string> seen;
		top.insert("metrics");
		if (j.contains("metrics")) {
			auto& m = j.at("metrics");
			READ(m, seen, cfg.sendMetrics, "send_metrics");
			READ(m, seen, cfg.metricsProjectName, "project");
			READ(m, seen, cfg.metricsGroupName, "group");
			READ(m, seen, cfg.metricsRunName, "run");
			READ(m, seen, cfg.skillTrackerEnabled, "skill_tracker");
			READ(m, seen, cfg.skillTimestepsPerVersion, "skill_timesteps_per_version");
			READ(m, seen, cfg.skillMaxVersions, "skill_max_versions");
			READ(m, seen, cfg.skillUpdateInterval, "skill_update_interval");
			CheckUnknown(m, seen, "metrics");
		}
	}
	top.insert("_comment");
	CheckUnknown(j, top, "Wurzel");

	// Plausibilität
	if (cfg.maxPlayers < 1 || cfg.maxPlayers > 4)
		RG_ERR_CLOSE("env.max_players muss zwischen 1 und 4 liegen");
	if (cfg.actionStackSize < 1)
		RG_ERR_CLOSE("env.action_stack_size muss >= 1 sein");
	float mixSum = cfg.modeMix[0] + cfg.modeMix[1] + cfg.modeMix[2];
	if (mixSum <= 0)
		RG_ERR_CLOSE("env.mode_mix darf nicht komplett 0 sein");
	for (int i = 0; i < 3; i++)
		if (cfg.modeMix[i] > 0 && (i + 1) > cfg.maxPlayers)
			RG_ERR_CLOSE("env.mode_mix aktiviert " << (i + 1) << "v" << (i + 1)
				<< ", aber max_players ist " << cfg.maxPlayers);
	if (cfg.ppoMiniBatchSize > cfg.ppoBatchSize)
		RG_ERR_CLOSE("learner.ppo_mini_batch_size darf nicht größer als ppo_batch_size sein");
	if (cfg.device != "cuda" && cfg.device != "cpu" && cfg.device != "auto")
		RG_ERR_CLOSE("learner.device muss cuda, cpu oder auto sein");
	if (cfg.extraSteps < 0)
		RG_ERR_CLOSE("learner.extra_steps darf nicht negativ sein");
	if (cfg.expBufferIterations < 1)
		RG_ERR_CLOSE("learner.exp_buffer_iterations muss >= 1 sein");

	return cfg;
}

std::string TrainConfig::ToJSONString() const {
	json j;
	j["env"] = {
		{ "tick_skip", tickSkip }, { "max_players", maxPlayers },
		{ "action_stack_size", actionStackSize },
		{ "no_touch_timeout_secs", noTouchTimeoutSecs }, { "game_timeout_secs", gameTimeoutSecs },
		{ "mode_mix", { modeMix[0], modeMix[1], modeMix[2] } },
		{ "shuffle_slots", shuffleSlots }, { "seed_envs", seedEnvs },
	};
	j["rewards"] = {
		{ "goal", rewards.goal }, { "concede", rewards.concede },
		{ "demo", rewards.demo }, { "demoed", rewards.demoed },
		{ "touch_ball_to_goal_accel", rewards.touchBallToGoalAccel },
		{ "offensive_potential_krc", rewards.offensivePotential },
		{ "dist_weighted_align_krc", rewards.distWeightedAlign },
		{ "velocity_player_to_ball", rewards.velocityPlayerToBall },
		{ "save_boost", rewards.saveBoost }, { "in_air", rewards.inAir },
		{ "team_spirit", rewards.teamSpirit },
	};
	j["state_setters"] = {
		{ "kickoff", states.kickoff }, { "random", states.random }, { "aerial", states.aerial },
		{ "dribble", states.dribble }, { "wall_play", states.wallPlay },
		{ "recovery", states.recovery }, { "defense", states.defense },
	};
	j["learner"] = {
		{ "num_threads", numThreads }, { "num_games_per_thread", numGamesPerThread },
		{ "collection_during_learn", collectionDuringLearn }, { "device", device },
		{ "timestep_limit", timestepLimit }, { "extra_steps", extraSteps }, { "save_on_exit", saveOnExit },
		{ "timesteps_per_iteration", timestepsPerIteration },
		{ "timesteps_per_save", timestepsPerSave }, { "checkpoints_to_keep", checkpointsToKeep },
		{ "checkpoint_folder", checkpointFolder },
		{ "policy_layer_sizes", policyLayerSizes }, { "critic_layer_sizes", criticLayerSizes },
		{ "ppo_epochs", ppoEpochs }, { "exp_buffer_iterations", expBufferIterations },
		{ "ppo_batch_size", ppoBatchSize },
		{ "ppo_mini_batch_size", ppoMiniBatchSize }, { "ent_coef", entCoef },
		{ "clip_range", clipRange }, { "policy_lr", policyLR }, { "critic_lr", criticLR },
		{ "gae_lambda", gaeLambda }, { "gae_gamma", gaeGamma }, { "random_seed", randomSeed },
	};
	j["metrics"] = {
		{ "send_metrics", sendMetrics }, { "project", metricsProjectName },
		{ "group", metricsGroupName }, { "run", metricsRunName },
		{ "skill_tracker", skillTrackerEnabled },
		{ "skill_timesteps_per_version", skillTimestepsPerVersion },
		{ "skill_max_versions", skillMaxVersions },
		{ "skill_update_interval", skillUpdateInterval },
	};
	return j.dump(2);
}

RLGPC::LearnerConfig MakeLearnerConfig(const TrainConfig& cfg) {
	RLGPC::LearnerConfig lc = {};
	lc.numThreads = cfg.numThreads;
	lc.numGamesPerThread = cfg.numGamesPerThread;
	lc.collectionDuringLearn = cfg.collectionDuringLearn;
	lc.deviceType = cfg.device == "cuda" ? RLGPC::LearnerDeviceType::GPU_CUDA
		: (cfg.device == "cpu" ? RLGPC::LearnerDeviceType::CPU : RLGPC::LearnerDeviceType::AUTO);
	lc.timestepLimit = cfg.timestepLimit;
	lc.timestepsPerIteration = cfg.timestepsPerIteration;
	lc.timestepsPerSave = cfg.timestepsPerSave;
	lc.checkpointsToKeep = cfg.checkpointsToKeep;
	lc.checkpointSaveFolder = cfg.checkpointFolder;
	lc.checkpointLoadFolder = cfg.checkpointFolder;
	lc.randomSeed = cfg.randomSeed;
	lc.gaeLambda = cfg.gaeLambda;
	lc.gaeGamma = cfg.gaeGamma;
	lc.expBufferSize = cfg.timestepsPerIteration * cfg.expBufferIterations;

	lc.ppo.policyLayerSizes = cfg.policyLayerSizes;
	lc.ppo.criticLayerSizes = cfg.criticLayerSizes;
	lc.ppo.epochs = cfg.ppoEpochs;
	lc.ppo.batchSize = cfg.ppoBatchSize;
	lc.ppo.miniBatchSize = cfg.ppoMiniBatchSize;
	lc.ppo.entCoef = cfg.entCoef;
	lc.ppo.clipRange = cfg.clipRange;
	lc.ppo.policyLR = cfg.policyLR;
	lc.ppo.criticLR = cfg.criticLR;

	lc.sendMetrics = cfg.sendMetrics;
	lc.metricsProjectName = cfg.metricsProjectName;
	lc.metricsGroupName = cfg.metricsGroupName;
	lc.metricsRunName = cfg.metricsRunName;

	lc.skillTrackerConfig.enabled = cfg.skillTrackerEnabled;
	lc.skillTrackerConfig.timestepsPerVersion = cfg.skillTimestepsPerVersion;
	lc.skillTrackerConfig.maxVersions = cfg.skillMaxVersions;
	lc.skillTrackerConfig.updateInterval = cfg.skillUpdateInterval;
	lc.skillTrackerConfig.perModeRatings = true;
	return lc;
}

} // namespace RLbot
