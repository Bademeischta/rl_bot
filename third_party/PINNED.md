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
git -C third_party/RLGymPPO_CPP apply ../../third_party/patches/rlgympppo_cpp_speed.patch
```

Die Patches sind ein Stapel: `rlgympppo_cpp_speed.patch` ändert Zeilen, die der Truncation-Patch
eingeführt hat. `apply_patches.ps1` bestimmt den Stand deshalb von hinten (lässt sich Patch k
umkehren, sind er und alle davor angewendet). Eigene Änderungen am Klon werden mit
`python tools/export_upstream_patch.py` in den letzten Patch geschrieben (Diff gegen gepinnten
Commit plus die Patches davor).

| Patch | Betrifft | Zweck |
|---|---|---|
| `rlgympppo_cpp_gcc_compat.patch` | `Util/Timer.h`, `Util/gradscaler.hpp` | GCC-13-Kompatibilität (Linux-CPU-Build der Tests in der Cloud-Session); auf MSVC ohne Wirkung |
| `rlgympppo_cpp_truncation.patch` | `TerminalCondition.h`, `Match.{h,cpp}`, `Gym.{h,cpp}`, `GameInst.cpp`, `ThreadAgent.cpp`, `ThreadAgentManager.{h,cpp}`, `TorchFuncs.{h,cpp}`, `Learner.{h,cpp}` | **Audit K1:** Zeitlimits als Truncation. `TerminalCondition::IsTruncation()`, `Gym::StepResult::truncated`, `ThreadAgent` speichert `done = done && !truncated`; `GameInst::Step` sichert die letzte Beobachtung einer beendeten Episode in `StepResult::finalObs`, bevor `obs` die Reset-Beobachtung wird, und `ThreadAgent` legt sie in `nextStates` ab (Review R4; die erste Fassung las hier die Reset-Beobachtung, AUDIT.md §7.2b); `ComputeGAE` bootstrappt an jedem `truncated`-Step mit `V(nextStates[step])` (behebt nebenbei das Bootstrapping an den Grenzen verketteter Trajektorien, AUDIT.md §7.2). Definiert `RLGSC_HAS_TRUNCATION` und `RLGSC_HAS_FINAL_OBS`; das Wurzel-CMake bricht ab, wenn der Patch fehlt oder in der alten Fassung angewendet ist (`tools\apply_patches.ps1 -Reset`). Report-Größen `Truncated Steps`, `Timeout Truncations`, `Trunc Bootstrap Reset Share`, `Trunc Bootstrap V Final/Reset/Diff`; Diagnose-Hook `Learner::truncationDiagnosticsCallback` für Tests |
| `rlgympppo_cpp_speed.patch` | `ThreadAgent.{h,cpp}`, `ThreadAgentManager.{h,cpp}`, `Learner.cpp`, `PPOLearner.cpp`, `TorchFuncs.{h,cpp}` (wächst mit AUDIT.md §9) | **Geschwindigkeit, AUDIT.md §9.** G1: Zeitaufschlüsselung je Iteration in `metrics.csv` (`Infer Call Time`, `Traj Append Time`, `Obs Tensor Time`, `Collect Concat Time`, `Add Experience Time` mit `Exp *`, `PPO Shuffle/Minibatch/Optim/Param Copy Time`, `Empty Cache Time`, `Prev * Time`); `RLBOT_PROFILE_SYNC=1` wartet an den Phasengrenzen auf die GPU (nur zum Profilen). G2: Sammel-Threads schreiben Schritte in einfache Arrays (`RawTrajectory`), `CollectTimesteps` fügt sie per memcpy zusammen (bitgleich zum alten Pfad, Prüfmodus `LearnerConfig::verifyTrajectories`); `ThreadAgent`/`ThreadAgentManager` exportiert, `set_num_interop_threads` nur einmal je Prozess (mehrere Learner in den Tests). G3: Warte-Schleifen der Sammel-Threads prüfen `shouldRun` (vorher hing `StopAgents()` am Sammel-Limit, docs/phase0_results.md §3). G4: `LearnerConfig::expBufferOnDevice` hält den Experience-Puffer im GPU-Speicher (Shuffle-Gather auf der GPU, keine Kopie je Minibatch, bitgleiche Ergebnisse); mehrere Learner je Prozess (eingebetteter Interpreter mit Zähler), `PPOLearner`/`ExperienceBuffer` exportiert. G5: `LearnerConfig::inferDuringLearn` (mit `collectionDuringLearn`): Sammel-Threads inferieren mit eigener Policy-Kopie (`Learner::inferPolicy`, nach jeder Lernphase per `SyncInferPolicy()` unter exklusiver Sperre aktualisiert) auf eigenen CUDA-Streams weiter, während PPO lernt; `collectLimitFactor` (Upstream 1,5) begrenzt den Vorlauf; Metrik `Steps Collected During Learn`, `Infer Policy Sync Time`. G6: `LearnerConfig::tf32` schaltet cuBLAS/cuDNN auf TF32 (nur ein, nie aus). G7: `ppo.autocastLearn` rechnet in BF16 ohne Grad-Scaler (Upstream clippte die mit 2^16 skalierten Gradienten vor dem Zurückskalieren). G8: `learnerHighPriorityStream` legt die GPU-Arbeit des Lern-Threads mit `inferDuringLearn` auf einen Stream hoher Priorität. B3: `RG_AUTOCAST_ON/OFF` (`FrameworkTorch.h`) mit `set_autocast_enabled`/`set_autocast_dtype` und Gerätetyp statt der veralteten Wrapper (keine Deprecation-Warnungen je Minibatch) |
| `libtorch_cuda_cmake_no_enable_language.patch` | libtorch cu128 `Caffe2/public/cuda.cmake` (**nicht** RLGymPPO_CPP) | nvcc 12.8 akzeptiert MSVC 14.51 nicht; `-DRLBOT_SKIP_CUDA_LANGUAGE=ON` überspringt `enable_language(CUDA)`. Anwenden mit `tools\patch_libtorch_cuda.ps1` |
