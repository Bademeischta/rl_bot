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

	if (std::filesystem::exists(g_meshDir)) {
		RocketSim::Init(g_meshDir);
		g_arenaReady = true;
	} else {
		std::cout << "WARNUNG: Meshes nicht gefunden unter '" << g_meshDir
		          << "', Arena-Tests werden übersprungen\n";
	}

	int failed = 0, passed = 0;
	for (auto& test : AllTests()) {
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
	          << AllTests().size() << " gesamt)\n";
	return failed == 0 ? 0 : 1;
}
