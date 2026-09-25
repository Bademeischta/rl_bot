cmake_minimum_required(VERSION 3.20)
project(RLbot CXX)

# Alles in einen Ordner, damit RLGymPPO_CPP.dll und die torch-DLLs neben den .exe liegen
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

set(UPSTREAM "${CMAKE_SOURCE_DIR}/third_party/RLGymPPO_CPP/RLGymPPO_CPP")
if(NOT EXISTS "${UPSTREAM}/CMakeLists.txt")
    message(FATAL_ERROR "third_party/RLGymPPO_CPP fehlt - siehe third_party/PINNED.md")
endif()

# Upstream unverändert einbinden (Torch_DIR/CMAKE_PREFIX_PATH kommt von bench/cpp/build.ps1)
add_subdirectory("${UPSTREAM}" "${CMAKE_BINARY_DIR}/RLGymPPO_CPP")

set(JSON_INCLUDE "${UPSTREAM}/libsrc/json")

# --- Environment-Bausteine (Rewards, Obs, State-Setter) ---
add_library(rlbot_env STATIC
    env/cpp/Rewards.cpp
    env/cpp/Obs.cpp
    env/cpp/StateSetters.cpp
)
set_target_properties(rlbot_env PROPERTIES CXX_STANDARD 20)
target_include_directories(rlbot_env PUBLIC "${CMAKE_SOURCE_DIR}")
target_link_libraries(rlbot_env PUBLIC RLGymSim_CPP)

# --- Trainer ---
add_executable(train_bot train/cpp/main.cpp train/cpp/Config.cpp train/cpp/EnvFactory.cpp train/cpp/Metrics.cpp)
set_target_properties(train_bot PROPERTIES CXX_STANDARD 20)
target_include_directories(train_bot PRIVATE "${JSON_INCLUDE}")
target_link_libraries(train_bot PRIVATE rlbot_env RLGymPPO_CPP)

# --- Phase-0-Benchmark ---
add_executable(bench_cpp_sps bench/cpp/main.cpp)
set_target_properties(bench_cpp_sps PROPERTIES CXX_STANDARD 20)
target_link_libraries(bench_cpp_sps PRIVATE RLGymPPO_CPP)

# --- Unit-Tests (eigene kleine Assert-Bibliothek, kein Framework nötig) ---
add_executable(rlbot_tests
    tests/cpp/test_main.cpp
    tests/cpp/test_rewards.cpp
    tests/cpp/test_obs.cpp
    tests/cpp/test_state_setters.cpp
    tests/cpp/test_config.cpp
    tests/cpp/test_metrics.cpp
    tests/cpp/test_env.cpp
    train/cpp/Config.cpp
    train/cpp/EnvFactory.cpp
    train/cpp/Metrics.cpp
)
set_target_properties(rlbot_tests PROPERTIES CXX_STANDARD 20)
target_include_directories(rlbot_tests PRIVATE "${JSON_INCLUDE}")
target_link_libraries(rlbot_tests PRIVATE rlbot_env RLGymPPO_CPP)

# --- Obs-Dump für den Paritätstest gegen die Python-Implementierung ---
add_executable(dump_obs tools/cpp/dump_obs.cpp)
set_target_properties(dump_obs PROPERTIES CXX_STANDARD 20)
target_include_directories(dump_obs PRIVATE "${JSON_INCLUDE}")
target_link_libraries(dump_obs PRIVATE rlbot_env)

# --- Policy-Dump: prüft, dass die Python-Inferenz im Deployment identisch rechnet ---
find_package(Torch REQUIRED)
add_executable(dump_policy_actions tools/cpp/dump_policy_actions.cpp)
set_target_properties(dump_policy_actions PROPERTIES CXX_STANDARD 20)
target_include_directories(dump_policy_actions PRIVATE "${JSON_INCLUDE}")
target_link_libraries(dump_policy_actions PRIVATE "${TORCH_LIBRARIES}")

# --- Duell zweier Checkpoints für die TrueSkill-Ladder ---
add_executable(duel eval/cpp/duel.cpp)
set_target_properties(duel PROPERTIES CXX_STANDARD 20)
target_include_directories(duel PRIVATE "${JSON_INCLUDE}" "${CMAKE_SOURCE_DIR}/eval/cpp")
target_link_libraries(duel PRIVATE rlbot_env "${TORCH_LIBRARIES}")

enable_testing()
add_test(NAME cpp_unit_tests COMMAND rlbot_tests)
