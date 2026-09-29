#include "test_util.h"

#include <RLGymSim_CPP/Framework.h>

#include <filesystem>

std::vector<TestCase>& AllTests() {
	static std::vector<TestCase> tests;
	return tests;
}

// Manche Tests brauchen eine echte Arena und damit die Collision-Meshes.
std::string g_meshDir = "collision_meshes";
bool g_arenaReady = false;

int main(int argc, char** argv) {
	if (argc > 1)
		g_meshDir = argv[1];
	// Optional: nur Tests, deren Name diesen Text enthält (z. B. "G3_" für eine Gegenprobe)
	std::string filter = argc > 2 ? argv[2] : "";

	if (std::filesystem::exists(g_meshDir)) {
		RocketSim::Init(g_meshDir);
		g_arenaReady = true;
	} else {
		std::cout << "WARNUNG: Meshes nicht gefunden unter '" << g_meshDir
		          << "', Arena-Tests werden übersprungen\n";
	}

	int failed = 0, passed = 0;
	size_t selected = 0;
	for (auto& test : AllTests()) {
		if (!filter.empty() && test.name.find(filter) == std::string::npos)
			continue;
		selected++;
		try {
			test.fn();
			std::cout << "[  OK  ] " << test.name << "\n";
			passed++;
		} catch (std::exception& e) {
			std::cout << "[ FAIL ] " << test.name << "\n         " << e.what() << "\n";
			failed++;
		}
	}
	std::cout << "\n" << passed << " bestanden, " << failed << " fehlgeschlagen ("
	          << selected << " gesamt)\n";
	return failed == 0 ? 0 : 1;
}
