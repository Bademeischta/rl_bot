# Wendet third_party/patches/libtorch_cuda_cmake_no_enable_language.patch auf ein frisch
# entpacktes libtorch cu128 an (idempotent).
#
# Hintergrund: libtorch ruft in Caffe2/public/cuda.cmake unbedingt enable_language(CUDA) auf.
# nvcc 12.8 akzeptiert MSVC 14.51 (VS 2026) nicht als Host-Compiler und cudafe++ stürzt ab
# (ACCESS_VIOLATION), auch mit -allow-unsupported-compiler. Wir kompilieren aber keine
# .cu-Dateien, sondern linken nur die vorgebaute torch_cuda.dll. Der Patch macht den Aufruf
# abschaltbar über -DRLBOT_SKIP_CUDA_LANGUAGE=ON (siehe bench/cpp/build.ps1).
$ErrorActionPreference = "Stop"
$f = "$PSScriptRoot\..\third_party\libtorch_cu128\libtorch\share\cmake\Caffe2\public\cuda.cmake"
if (-not (Test-Path $f)) { throw "libtorch cu128 nicht gefunden: $f" }

$s = Get-Content $f -Raw
if ($s -match "RLBOT_SKIP_CUDA_LANGUAGE") { Write-Host "Patch ist bereits angewendet."; exit 0 }
if (-not (Test-Path "$f.orig")) { Copy-Item $f "$f.orig" }

$old = "enable_language(CUDA)`n"
$new = @"
if(RLBOT_SKIP_CUDA_LANGUAGE)
  message(STATUS "RLbot: skipping enable_language(CUDA)")
else()
  enable_language(CUDA)
endif()

"@
if ($s.IndexOf($old) -lt 0) { throw "enable_language(CUDA) nicht gefunden - libtorch-Version geprueft?" }
$s = $s.Replace($old, $new)

$old2 = "find_package(CUDAToolkit REQUIRED)`n"
$new2 = $old2 + @"
if(RLBOT_SKIP_CUDA_LANGUAGE)
  set(CMAKE_CUDA_COMPILER_VERSION "`${CUDAToolkit_VERSION}")
endif()

"@
$s = $s.Replace($old2, $new2)
Set-Content -Path $f -Value $s -NoNewline -Encoding utf8
Write-Host "Patch angewendet: $f"
