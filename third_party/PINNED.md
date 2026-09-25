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

Der Build läuft über das Wurzel-`CMakeLists.txt` (`bench/cpp/build.ps1`), das den Upstream mit
`add_subdirectory` einbindet.

## Patches am Upstream-Code (`patches/`)

Alle Patches sind gegen den gepinnten Commit `ee4cc56` erzeugt (`git diff` im Klon) und werden
von `tools/apply_patches.ps1` idempotent angewendet; `bench/cpp/build.ps1` ruft das Skript vor
jedem Build auf. Prüfen ohne anzuwenden: `tools\apply_patches.ps1 -Check`. Von Hand:

```bash
git -C third_party/RLGymPPO_CPP apply ../../third_party/patches/rlgympppo_cpp_gcc_compat.patch
git -C third_party/RLGymPPO_CPP apply ../../third_party/patches/rlgympppo_cpp_truncation.patch
```

| Patch | Betrifft | Zweck |
|---|---|---|
| `rlgympppo_cpp_gcc_compat.patch` | `Util/Timer.h`, `Util/gradscaler.hpp` | GCC-13-Kompatibilität (Linux-CPU-Build der Tests in der Cloud-Session); auf MSVC ohne Wirkung |
| `rlgympppo_cpp_truncation.patch` | `TerminalCondition.h`, `Match.{h,cpp}`, `Gym.{h,cpp}`, `ThreadAgent.cpp`, `ThreadAgentManager.cpp`, `TorchFuncs.{h,cpp}`, `Learner.cpp` | **Audit K1:** Zeitlimits als Truncation. `TerminalCondition::IsTruncation()`, `Gym::StepResult::truncated`, `ThreadAgent` speichert `done = done && !truncated` und legt für beendete Episoden die letzte Beobachtung (nicht die Reset-Beobachtung) in `nextStates` ab; `ComputeGAE` bootstrappt an jedem `truncated`-Step mit `V(nextStates[step])` (behebt nebenbei das Bootstrapping an den Grenzen verketteter Trajektorien, AUDIT.md §7.2). Definiert `RLGSC_HAS_TRUNCATION`; das Wurzel-CMake bricht ab, wenn der Patch fehlt. Neue Report-Größe `Truncated Steps` |
| `libtorch_cuda_cmake_no_enable_language.patch` | libtorch cu128 `Caffe2/public/cuda.cmake` (**nicht** RLGymPPO_CPP) | nvcc 12.8 akzeptiert MSVC 14.51 nicht; `-DRLBOT_SKIP_CUDA_LANGUAGE=ON` überspringt `enable_language(CUDA)`. Anwenden mit `tools\patch_libtorch_cuda.ps1` |
