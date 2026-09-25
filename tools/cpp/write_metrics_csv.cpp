// Schreibt eine metrics.csv mit dem echten Writer des Trainers (RLbot::MetricsCSVWriter), für den
// End-to-End-Test der Abbruchkriterien (Review-Befund R5, tests/test_experiments_tools.py):
// C++-Writer -> CSV -> tools/experiments/check_abort.py.
//
//   write_metrics_csv <out.csv> <iterationen> [--set <iteration> <spalte> <wert>]... [--drop <iteration> <spalte>]...
//
// Jede Iteration bekommt plausible Werte wie im Hauptlauf; --set überschreibt einen Wert (auch
// "nan", "inf", "-inf"), --drop lässt den Schlüssel in dieser Iteration weg (-> leeres Feld).
#include "../../train/cpp/Metrics.h"

#include <iostream>
#include <map>
#include <string>
#include <utility>

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "usage: write_metrics_csv <out.csv> <iterationen> [--set <i> <spalte> <wert>]... [--drop <i> <spalte>]...\n";
		return 2;
	}
	std::filesystem::path out = argv[1];
	int iterations = std::stoi(argv[2]);

	std::map<std::pair<int, std::string>, double> sets;
	std::map<std::pair<int, std::string>, bool> drops;
	for (int a = 3; a < argc; a++) {
		std::string arg = argv[a];
		if (arg == "--set" && a + 3 < argc) {
			sets[{ std::stoi(argv[a + 1]), argv[a + 2] }] = std::stod(argv[a + 3]);   // stod kennt nan/inf
			a += 3;
		} else if (arg == "--drop" && a + 2 < argc) {
			drops[{ std::stoi(argv[a + 1]), argv[a + 2] }] = true;
			a += 2;
		} else {
			std::cerr << "Unbekanntes Argument: " << arg << "\n";
			return 2;
		}
	}

	RLbot::MetricsCSVWriter writer(out);
	for (int i = 0; i < iterations; i++) {
		RLGPC::Report report;
		report["Cumulative Timesteps"] = 3907335040.0 + (i + 1) * 100000.0;
		report["Timesteps Collected"] = 100000;
		report["Overall Steps/Second"] = 68000;
		report["Policy Entropy"] = 3.58;
		report["Value Function Loss"] = 0.25;
		report["Mean KL Divergence"] = 0.0027;
		report["Average Episode Reward"] = 2049.0;
		report["Avg Advantage"] = 0.4;
		report["Avg Val Target"] = 10.0;
		for (auto& [key, value] : sets)
			if (key.first == i) report[key.second] = value;
		for (auto& [key, drop] : drops)
			if (key.first == i) report.data.erase(key.second);
		writer.Append(report);
	}
	std::cout << "geschrieben: " << out.string() << " (" << iterations << " Iterationen)\n";
	return 0;
}
