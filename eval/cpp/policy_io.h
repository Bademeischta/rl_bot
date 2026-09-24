// Lädt eine mit torch::save gespeicherte Policy und leitet die Netzgröße aus den
// Gewichtsformen ab, damit kein separates Config-Argument nötig ist.
#pragma once

#include <torch/torch.h>
#include <torch/script.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace RLbot {

struct LoadedPolicy {
	torch::nn::Sequential seq = nullptr;
	int obsSize = 0;
	int actionCount = 0;
	std::vector<int> layerSizes;
};

inline LoadedPolicy LoadPolicy(const std::string& path, torch::Device device = torch::kCPU) {
	torch::jit::script::Module mod;
	try {
		mod = torch::jit::load(path, torch::kCPU);
	} catch (const std::exception& e) {
		throw std::runtime_error("Policy nicht ladbar (" + path + "): " + e.what());
	}

	// Parameter heißen "<index>.weight" / "<index>.bias" in Reihenfolge der Module
	std::vector<std::pair<int, at::Tensor>> weights, biases;
	for (const auto& p : mod.named_parameters(true)) {
		int idx = std::stoi(p.name.substr(0, p.name.find('.')));
		(p.name.find("weight") != std::string::npos ? weights : biases).push_back({ idx, p.value });
	}
	if (weights.empty())
		throw std::runtime_error("Policy enthält keine Gewichte: " + path);

	auto byIndex = [](auto& a, auto& b) { return a.first < b.first; };
	std::sort(weights.begin(), weights.end(), byIndex);
	std::sort(biases.begin(), biases.end(), byIndex);

	LoadedPolicy out;
	out.obsSize = (int)weights.front().second.size(1);
	out.actionCount = (int)weights.back().second.size(0);
	for (size_t i = 0; i + 1 < weights.size(); i++)
		out.layerSizes.push_back((int)weights[i].second.size(0));

	out.seq = torch::nn::Sequential();
	out.seq->push_back(torch::nn::Linear(out.obsSize, out.layerSizes[0]));
	out.seq->push_back(torch::nn::ReLU());
	for (size_t i = 1; i < out.layerSizes.size(); i++) {
		out.seq->push_back(torch::nn::Linear(out.layerSizes[i - 1], out.layerSizes[i]));
		out.seq->push_back(torch::nn::ReLU());
	}
	out.seq->push_back(torch::nn::Linear(out.layerSizes.back(), out.actionCount));

	{
		torch::NoGradGuard noGrad;
		auto params = out.seq->parameters();
		// Sequential-Parameter kommen als weight/bias-Paare je Linear-Schicht
		if (params.size() != weights.size() + biases.size())
			throw std::runtime_error("Parameteranzahl passt nicht zur abgeleiteten Architektur");
		for (size_t i = 0; i < weights.size(); i++) {
			params[i * 2].copy_(weights[i].second);
			params[i * 2 + 1].copy_(biases[i].second);
		}
	}
	out.seq->to(device);
	out.seq->eval();
	return out;
}

// Rechenweg wie RLGymPPO_CPP DiscretePolicy
inline at::Tensor PolicyProbs(torch::nn::Sequential& seq, at::Tensor obs, float temperature = 1.f) {
	constexpr float ACTION_MIN_PROB = 1e-11f;
	auto probs = torch::softmax(seq->forward(obs) / temperature, -1);
	return torch::clamp(probs, ACTION_MIN_PROB, 1);
}

} // namespace RLbot
