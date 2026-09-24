# Gepinnte Fremdabhängigkeiten

Klone liegen unter `third_party/` und sind gitignored. Wiederherstellen mit den Befehlen unten.

| Komponente | Quelle | Version / Commit | Zweck |
|---|---|---|---|
| RLGymPPO_CPP | https://github.com/ZealanL/RLGymPPO_CPP | `ee4cc56fc8e43758cc898fdba92a9173a94e52f2` (2024-10-26) | C++-Learner (Plan A) |
| ├ RocketSim | im Repo eingebettet (`RLGymSim_CPP/RocketSim`) | wie oben | Physik |
| ├ pybind11 | Submodul | `8b48ff878c168b51fe5ef7b8c728815b9e1a9857` | Python-Embed (Metrics) |
| └ RLBotCPP | Submodul | `5b1d27f5279061859a80156a70750adc0ba3d475` | Deploy (später) |
| libtorch CPU | https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.11.0%2Bcpu.zip | 2.11.0+cpu | Spike 5a → `third_party/libtorch_cpu/` |
| libtorch cu128 | https://download.pytorch.org/libtorch/cu128/libtorch-win-shared-with-deps-2.11.0%2Bcu128.zip | 2.11.0+cu128 | Spike 5b → `third_party/libtorch_cu128/` |

```bash
git clone --recursive https://github.com/ZealanL/RLGymPPO_CPP.git third_party/RLGymPPO_CPP
git -C third_party/RLGymPPO_CPP checkout ee4cc56fc8e43758cc898fdba92a9173a94e52f2
git -C third_party/RLGymPPO_CPP submodule update --init --recursive
```

Toolchain: VS Build Tools 2026 (MSVC 14.51), CMake 4.2.3 + Ninja 1.13.2 aus den Build Tools, CUDA Toolkit 12.8 (V12.8.61).

Patches am Upstream-Code: keine (siehe `patches/`). Der Benchmark-Build bindet Upstream über `bench/cpp/CMakeLists.txt` unverändert ein.
