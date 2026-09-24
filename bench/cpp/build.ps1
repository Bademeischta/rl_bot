# Baut die C++-Targets gegen CPU- oder cu128-libtorch.
#   powershell -File bench\cpp\build.ps1 -Flavor cu128
#   powershell -File bench\cpp\build.ps1 -Flavor cpu -Target rlbot_tests
param(
    [ValidateSet("cpu", "cu128")][string]$Flavor = "cu128",
    [string]$Target = "all",
    [string]$ExtraCxxFlags = "",
    # Eigenes Build-Verzeichnis, z.B. um eine Variante zu bauen, während die andere läuft
    [string]$BuildSuffix = ""
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path "$PSScriptRoot\..\.."
$VS = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
$CMakeBin = "$VS\Common7\IDE\CommonExtensions\Microsoft\CMake"

# MSVC-Umgebung (x64) in diese PowerShell-Session übernehmen
cmd /c "`"$VS\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") { Set-Item "env:$($matches[1])" $matches[2] }
}
$env:PATH = "$CMakeBin\CMake\bin;$CMakeBin\Ninja;$env:PATH"

$Torch = "$Root\third_party\libtorch_$Flavor\libtorch"
if (-not (Test-Path $Torch)) { throw "libtorch fehlt: $Torch (siehe third_party\PINNED.md)" }
$Build = "$Root\build\cpp_$Flavor$BuildSuffix"
$Py = (& "$Root\.venv\Scripts\python.exe" -c "import sys; print(sys.base_prefix)").Trim()

$cmakeArgs = @(
    "-S", $Root, "-B", $Build, "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_PREFIX_PATH=$Torch",
    "-DPython_ROOT_DIR=$Py",
    "-DPYTHON_EXECUTABLE=$Py\python.exe",
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    "-DCMAKE_CXX_FLAGS=/EHsc /utf-8 $ExtraCxxFlags"
)
if ($Flavor -eq "cu128") {
    $cmakeArgs += "-DCUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    $cmakeArgs += "-DTORCH_CUDA_ARCH_LIST=12.0"
    # nvcc 12.8 lehnt MSVC 14.51 (VS 2026) ab und cudafe++ stürzt selbst mit
    # -allow-unsupported-compiler ab. Das Projekt hat keine .cu-Dateien, nur die vorgebaute
    # torch_cuda.dll wird gelinkt, deshalb wird die CUDA-Sprache übersprungen.
    # Dazu der Patch third_party/patches/libtorch_cuda_cmake_no_enable_language.patch
    # (anwenden mit tools\patch_libtorch_cuda.ps1).
    $cmakeArgs += "-DRLBOT_SKIP_CUDA_LANGUAGE=ON"
}

cmake @cmakeArgs
if ($LASTEXITCODE) { exit $LASTEXITCODE }

if ($Target -eq "all") {
    cmake --build $Build
} else {
    cmake --build $Build --target $Target
}
exit $LASTEXITCODE
