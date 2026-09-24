// Lädt eine trainierte Policy (PPO_POLICY.lt) und schreibt für gegebene Obs-Vektoren die
// Aktionswahrscheinlichkeiten und die deterministische Aktion.
//
//   dump_policy_actions.exe <policy.lt> <obs_size> <action_count> <layers,csv> <obs.json> <out.json>
//
// Rechenweg identisch zu RLGymPPO_CPP DiscretePolicy: Sequential(Linear+ReLU..., Linear),
// Softmax, clamp auf ACTION_MIN_PROB, argmax für deterministisch.
// tests/test_policy_parity.py vergleicht das mit deploy/policy.py.
#include <torch/torch.h>
#include <torch/script.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

using nlohmann::json;

constexpr float ACTION_MIN_PROB = 1e-11f;

int main(int argc, char** argv) {
	if (argc < 7) {
		std::cerr << "usage: dump_policy_actions <policy.lt> <obs_size> <action_count> "
		             "<layers,csv> <obs.json> <out.json>\n";
		return 2;
	}
	std::string policyPath = argv[1];
	int obsSize = std::stoi(argv[2]);
	int actionCount = std::stoi(argv[3]);

	std::vector<int> layers;
	{
		std::stringstream ss(argv[4]);
		std::string item;
		while (std::getline(ss, item, ','))
			if (!item.empty()) layers.push_back(std::stoi(item));
	}

	torch::NoGradGuard noGrad;

	torch::nn::Sequential seq;
	seq->push_back(torch::nn::Linear(obsSize, layers[0]));
	seq->push_back(torch::nn::ReLU());
	for (size_t i = 1; i < layers.size(); i++) {
		seq->push_back(torch::nn::Linear(layers[i - 1], layers[i]));
		seq->push_back(torch::nn::ReLU());
	}
	seq->push_back(torch::nn::Linear(layers.back(), actionCount));

	{
		auto streamIn = std::ifstream(policyPath, std::ios::binary);
		if (!streamIn.good()) {
			std::cerr << "Policy nicht lesbar: " << policyPath << "\n";
			return 1;
		}
		torch::load(seq, streamIn, torch::kCPU);
	}

	json obsJson;
	{
		std::ifstream fIn(argv[5]);
		if (!fIn.good()) {
			std::cerr << "Obs-Datei nicht lesbar: " << argv[5] << "\n";
			return 1;
		}
		fIn >> obsJson;
	}

	auto rows = obsJson.get<std::vector<std::vector<float>>>();
	auto input = torch::zeros({ (int64_t)rows.size(), obsSize });
	for (size_t i = 0; i < rows.size(); i++) {
		if ((int)rows[i].size() != obsSize) {
			std::cerr << "Obs " << i << " hat Länge " << rows[i].size()
			          << ", erwartet " << obsSize << "\n";
			return 1;
		}
		input[i] = torch::from_blob(rows[i].data(), { obsSize }).clone();
	}

	auto logits = seq->forward(input);
	auto probs = torch::softmax(logits, -1);
	probs = torch::clamp(probs, ACTION_MIN_PROB, 1);
	auto best = probs.argmax(1);

	json out;
	out["probs"] = json::array();
	out["actions"] = json::array();
	for (int64_t i = 0; i < (int64_t)rows.size(); i++) {
		json row = json::array();
		for (int a = 0; a < actionCount; a++)
			row.push_back(probs[i][a].item<float>());
		out["probs"].push_back(row);
		out["actions"].push_back(best[i].item<int64_t>());
	}

	std::ofstream fOut(argv[6]);
	fOut << out.dump();
	std::cout << "Geschrieben: " << argv[6] << " (" << rows.size() << " Obs)\n";
	return 0;
}
