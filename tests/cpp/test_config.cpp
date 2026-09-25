// Prüft das Laden der Trainings-Config und die Verteilung der Teamgrößen (Modus-Mix).
#include "test_util.h"

#include "train/cpp/Config.h"
#include "train/cpp/EnvFactory.h"

#include <fstream>
#include <map>

using namespace RLbot;

static std::filesystem::path WriteTempConfig(const std::string& content) {
	static int counter = 0;
	auto path = std::filesystem::temp_directory_path()
		/ ("rlbot_test_cfg_" + std::to_string(counter++) + ".json");
	std::ofstream(path) << content;
	return path;
}

// RG_ERR_CLOSE wirft eine runtime_error; das nutzen wir, um Fehlerfälle zu prüfen.
static bool LoadFails(const std::string& content) {
	auto path = WriteTempConfig(content);
	try {
		TrainConfig::FromFile(path.string());
		std::filesystem::remove(path);
		return false;
	} catch (const std::exception&) {
		std::filesystem::remove(path);
		return true;
	}
}

TEST(Config_liest_alle_Bereiche) {
	auto path = WriteTempConfig(R"({
		"env": { "tick_skip": 12, "max_players": 2, "action_stack_size": 3,
		         "mode_mix": [1.0, 3.0, 0.0], "no_touch_timeout_secs": 11.5 },
		"rewards": { "goal": 5.0, "offensive_potential_krc": 0.25, "team_spirit": 0.7 },
		"state_setters": { "kickoff": 2.0, "random": 0.0, "aerial": 1.5 },
		"learner": { "num_threads": 4, "device": "cpu", "policy_layer_sizes": [64, 32],
		             "ppo_batch_size": 1000, "ppo_mini_batch_size": 500, "gae_gamma": 0.99 },
		"metrics": { "send_metrics": false, "run": "test" }
	})");
	auto cfg = TrainConfig::FromFile(path.string());
	std::filesystem::remove(path);

	CHECK_EQ(cfg.tickSkip, 12);
	CHECK_EQ(cfg.maxPlayers, 2);
	CHECK_EQ(cfg.actionStackSize, 3);
	CHECK_NEAR(cfg.noTouchTimeoutSecs, 11.5, 1e-6);
	CHECK_NEAR(cfg.rewards.goal, 5.0, 1e-6);
	CHECK_NEAR(cfg.rewards.offensivePotential, 0.25, 1e-6);
	CHECK_NEAR(cfg.rewards.teamSpirit, 0.7, 1e-6);
	CHECK_NEAR(cfg.states.aerial, 1.5, 1e-6);
	CHECK_NEAR(cfg.states.random, 0.0, 1e-6);
	CHECK_EQ(cfg.numThreads, 4);
	CHECK_EQ(cfg.device, std::string("cpu"));
	CHECK_EQ((int)cfg.policyLayerSizes.size(), 2);
	CHECK_EQ(cfg.policyLayerSizes[0], 64);
	CHECK_EQ(cfg.metricsRunName, std::string("test"));
}

TEST(Config_behaelt_Standardwerte_bei_leerer_Datei) {
	auto path = WriteTempConfig("{}");
	auto cfg = TrainConfig::FromFile(path.string());
	std::filesystem::remove(path);

	TrainConfig defaults = {};
	CHECK_EQ(cfg.tickSkip, defaults.tickSkip);
	CHECK_NEAR(cfg.rewards.goal, defaults.rewards.goal, 1e-6);
	CHECK_EQ(cfg.numGamesPerThread, defaults.numGamesPerThread);
}

TEST(Config_meldet_Tippfehler_in_Feldnamen) {
	// Ein stillschweigend ignoriertes Feld wäre schlimmer als ein Abbruch: Der Lauf liefe
	// tagelang mit anderen Werten als gedacht.
	CHECK(LoadFails(R"({"rewards": {"gaol": 5.0}})"));
	CHECK(LoadFails(R"({"lerner": {"num_threads": 4}})"));
	CHECK(LoadFails(R"({"env": {"tickskip": 8}})"));
}

TEST(Config_prueft_Plausibilitaet) {
	CHECK(LoadFails(R"({"env": {"max_players": 0}})"));
	CHECK(LoadFails(R"({"env": {"max_players": 9}})"));
	CHECK(LoadFails(R"({"env": {"action_stack_size": 0}})"));
	CHECK(LoadFails(R"({"env": {"mode_mix": [0, 0, 0]}})"));
	CHECK(LoadFails(R"({"env": {"mode_mix": [1, 1]}})"));
	// 3v3 angefordert, aber Padding nur für 2 Spieler
	CHECK(LoadFails(R"({"env": {"max_players": 2, "mode_mix": [1, 0, 1]}})"));
	CHECK(LoadFails(R"({"learner": {"ppo_batch_size": 100, "ppo_mini_batch_size": 200}})"));
	CHECK(LoadFails(R"({"learner": {"device": "tpu"}})"));
	CHECK(LoadFails("kein json"));
}

TEST(Config_Roundtrip_ueber_JSON) {
	auto path = WriteTempConfig(R"({
		"env": { "tick_skip": 6, "mode_mix": [2.0, 1.0, 1.0], "max_players": 3 },
		"rewards": { "goal": 7.5, "in_air": 0.05 },
		"learner": { "num_threads": 12, "policy_layer_sizes": [128, 64] }
	})");
	auto cfg = TrainConfig::FromFile(path.string());
	std::filesystem::remove(path);

	auto path2 = WriteTempConfig(cfg.ToJSONString());
	auto cfg2 = TrainConfig::FromFile(path2.string());
	std::filesystem::remove(path2);

	CHECK_EQ(cfg2.tickSkip, cfg.tickSkip);
	CHECK_NEAR(cfg2.rewards.goal, cfg.rewards.goal, 1e-6);
	CHECK_NEAR(cfg2.rewards.inAir, cfg.rewards.inAir, 1e-6);
	CHECK_EQ(cfg2.numThreads, cfg.numThreads);
	CHECK_EQ((int)cfg2.policyLayerSizes.size(), 2);
	for (int i = 0; i < 3; i++)
		CHECK_NEAR(cfg2.modeMix[i], cfg.modeMix[i], 1e-6);
}

TEST(Config_wird_korrekt_in_LearnerConfig_uebertragen) {
	TrainConfig cfg = {};
	cfg.numThreads = 7;
	cfg.numGamesPerThread = 33;
	cfg.device = "cpu";
	cfg.timestepsPerIteration = 12345;
	cfg.ppoEpochs = 4;
	cfg.entCoef = 0.02f;
	cfg.policyLayerSizes = { 128, 64 };
	cfg.checkpointFolder = "runs/x/checkpoints";

	auto lc = MakeLearnerConfig(cfg);
	CHECK_EQ(lc.numThreads, 7);
	CHECK_EQ(lc.numGamesPerThread, 33);
	CHECK(lc.deviceType == RLGPC::LearnerDeviceType::CPU);
	CHECK_EQ((int)lc.timestepsPerIteration, 12345);
	CHECK_EQ((int)lc.expBufferSize, 12345 * 3);   // Puffer = drei Iterationen
	CHECK_EQ(lc.ppo.epochs, 4);
	CHECK_NEAR(lc.ppo.entCoef, 0.02, 1e-6);
	CHECK_EQ((int)lc.ppo.policyLayerSizes.size(), 2);
	// Laden und Speichern zeigen auf denselben Ordner, damit Läufe fortsetzbar sind
	CHECK_EQ(lc.checkpointLoadFolder.string(), lc.checkpointSaveFolder.string());

	cfg.device = "cuda";
	CHECK(MakeLearnerConfig(cfg).deviceType == RLGPC::LearnerDeviceType::GPU_CUDA);
	cfg.device = "auto";
	CHECK(MakeLearnerConfig(cfg).deviceType == RLGPC::LearnerDeviceType::AUTO);
}

// --- Modus-Mix (Phase 4) -------------------------------------------------

static std::map<int, int> CountTeamSizes(const float mix[3], int envCount) {
	TrainConfig cfg = {};
	cfg.maxPlayers = 3;
	for (int i = 0; i < 3; i++) cfg.modeMix[i] = mix[i];

	EnvFactory factory(cfg);
	std::map<int, int> counts;
	for (int i = 0; i < envCount; i++)
		counts[factory.TeamSizeForIndex(i)]++;
	return counts;
}

TEST(ModusMix_nur_1v1) {
	float mix[3] = { 1.f, 0.f, 0.f };
	auto counts = CountTeamSizes(mix, 256);
	CHECK_EQ(counts[1], 256);
	CHECK_EQ(counts[2], 0);
	CHECK_EQ(counts[3], 0);
}

TEST(ModusMix_gleichmaessig_ueber_drei_Modi) {
	float mix[3] = { 1.f, 1.f, 1.f };
	auto counts = CountTeamSizes(mix, 300);
	for (int teamSize = 1; teamSize <= 3; teamSize++)
		CHECK_NEAR(counts[teamSize] / 300.0, 1.0 / 3.0, 0.02);
}

TEST(ModusMix_gewichtet) {
	float mix[3] = { 6.f, 3.f, 1.f };   // 60 / 30 / 10 Prozent
	auto counts = CountTeamSizes(mix, 1000);
	CHECK_NEAR(counts[1] / 1000.0, 0.6, 0.02);
	CHECK_NEAR(counts[2] / 1000.0, 0.3, 0.02);
	CHECK_NEAR(counts[3] / 1000.0, 0.1, 0.02);
}

TEST(ModusMix_ist_deterministisch) {
	float mix[3] = { 2.f, 1.f, 1.f };
	TrainConfig cfg = {};
	cfg.maxPlayers = 3;
	for (int i = 0; i < 3; i++) cfg.modeMix[i] = mix[i];

	EnvFactory a(cfg), b(cfg);
	for (int i = 0; i < 100; i++)
		CHECK_EQ(a.TeamSizeForIndex(i), b.TeamSizeForIndex(i));
}

TEST(ModusMix_trifft_kleine_Anteile_auch_bei_wenigen_Envs) {
	// 5 Prozent 3v3 müssen bei 100 Envs auch wirklich vorkommen
	float mix[3] = { 0.75f, 0.20f, 0.05f };
	auto counts = CountTeamSizes(mix, 100);
	CHECK_GT(counts[3], 0);
	CHECK_GT(counts[2], 0);
	CHECK_NEAR(counts[1] / 100.0, 0.75, 0.05);
}

// --- Schalter aus dem Audit (H3 shuffle_slots, H6 seed_envs) ----------------

TEST(Config_shuffle_slots_und_seed_envs_haben_altes_Verhalten_als_Default) {
	auto path = WriteTempConfig("{}");
	auto cfg = TrainConfig::FromFile(path.string());
	std::filesystem::remove(path);
	CHECK(cfg.shuffleSlots);
	CHECK(cfg.seedEnvs);
}

TEST(Config_shuffle_slots_und_seed_envs_sind_lesbar_und_roundtrip_fest) {
	auto path = WriteTempConfig(R"({"env": {"shuffle_slots": false, "seed_envs": false}})");
	auto cfg = TrainConfig::FromFile(path.string());
	std::filesystem::remove(path);
	CHECK(!cfg.shuffleSlots);
	CHECK(!cfg.seedEnvs);

	auto path2 = WriteTempConfig(cfg.ToJSONString());
	auto cfg2 = TrainConfig::FromFile(path2.string());
	std::filesystem::remove(path2);
	CHECK(!cfg2.shuffleSlots);
	CHECK(!cfg2.seedEnvs);
}

// --- Experiment-Optionen (Stufe 3): extra_steps, save_on_exit ------------------

TEST(Config_extra_steps_und_save_on_exit_Default_ist_aus) {
	auto path = WriteTempConfig("{}");
	auto cfg = TrainConfig::FromFile(path.string());
	std::filesystem::remove(path);
	CHECK_EQ((int)cfg.extraSteps, 0);
	CHECK(!cfg.saveOnExit);
}

TEST(Config_extra_steps_und_save_on_exit_lesbar_und_geprueft) {
	auto path = WriteTempConfig(R"({"learner": {"extra_steps": 100000000, "save_on_exit": true}})");
	auto cfg = TrainConfig::FromFile(path.string());
	std::filesystem::remove(path);
	CHECK_EQ(cfg.extraSteps, (int64_t)100000000);
	CHECK(cfg.saveOnExit);

	auto path2 = WriteTempConfig(cfg.ToJSONString());
	auto cfg2 = TrainConfig::FromFile(path2.string());
	std::filesystem::remove(path2);
	CHECK_EQ(cfg2.extraSteps, (int64_t)100000000);
	CHECK(cfg2.saveOnExit);

	CHECK(LoadFails(R"({"learner": {"extra_steps": -5}})"));
}
