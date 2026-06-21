[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$Smoke,
    [switch]$Probe
)

$ErrorActionPreference = "Stop"

# Install and validate the pinned Isaac Sim 6 / ROS 2 Jazzy runtime.
# Global Python and ROS installations are not modified.

$PixiVersion = "0.70.2"
$PixiArchive = "pixi-x86_64-pc-windows-msvc.zip"
$PixiArchiveSha256 = "90bab8eb79031ae406119dc33ca24bc680a17b05364b96174c5d74a26563509c"
$IsaacLabCommit = "28a37cecdd433c22d9eabd6a5954add9f13a8951"
$LockSha256 = "9736a03f7b8b2b1d94d40285d0dc3508886cb38d2f04d9c885099ae50a31fcc5"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RuntimeRoot = if ($env:SO101_RUNTIME_ROOT) {
    [System.IO.Path]::GetFullPath($env:SO101_RUNTIME_ROOT)
} else {
    "D:\SO101\isaac6_ros"
}
$PixiRoot = Join-Path $RuntimeRoot ".pixi"
$ProjectPixi = Join-Path $ProjectRoot ".pixi"
$BinRoot = Join-Path $RuntimeRoot "bin"
$Pixi = Join-Path $BinRoot "pixi.exe"
$IsaacLabRoot = Join-Path $RuntimeRoot "IsaacLab"

function Require-Command {
    param([Parameter(Mandatory = $true)][string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $FilePath $($ArgumentList -join ' ')"
    }
}

if (-not $IsWindows -and $PSVersionTable.PSEdition -eq "Core") {
    throw "This script is for a Windows host"
}

Require-Command git
Require-Command jq
if (-not $CheckOnly) {
    Require-Command gh
}

Set-Location $ProjectRoot
foreach ($required in @(
    "pixi.toml",
    "pixi.lock",
    "configs\parity\runtime_manifest.mock.json",
    "configs\parity\replay_checkpoint.json"
)) {
    if (-not (Test-Path (Join-Path $ProjectRoot $required))) {
        throw "Required tracked file not found: $required"
    }
}

New-Item -ItemType Directory -Force -Path $BinRoot, $PixiRoot, (Join-Path $ProjectRoot "outputs\parity") | Out-Null

if (-not $CheckOnly -and -not (Test-Path $Pixi)) {
    $archivePath = Join-Path $BinRoot $PixiArchive
    if (Test-Path $archivePath) {
        Remove-Item -LiteralPath $archivePath
    }
    Invoke-Checked -FilePath gh -ArgumentList @(
        "release", "download", "v$PixiVersion",
        "--repo", "prefix-dev/pixi",
        "--pattern", $PixiArchive,
        "--dir", $BinRoot
    )

    $archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
    if ($archiveHash -ne $PixiArchiveSha256) {
        throw "Pixi archive hash mismatch: actual=$archiveHash expected=$PixiArchiveSha256"
    }

    $extractRoot = Join-Path $BinRoot "pixi-$PixiVersion"
    if (Test-Path $extractRoot) {
        Remove-Item -LiteralPath $extractRoot -Recurse
    }
    Expand-Archive -LiteralPath $archivePath -DestinationPath $extractRoot
    $extractedPixi = Get-ChildItem -Path $extractRoot -Recurse -Filter "pixi.exe" | Select-Object -First 1
    if (-not $extractedPixi) {
        throw "pixi.exe was not found in the Pixi archive"
    }
    Copy-Item -LiteralPath $extractedPixi.FullName -Destination $Pixi
}

if (-not (Test-Path $Pixi)) {
    throw "Pixi executable not found: $Pixi"
}
$actualPixiVersion = ((& $Pixi --version) -split "\s+")[-1]
if ($actualPixiVersion -ne $PixiVersion) {
    throw "Pixi version mismatch: actual=$actualPixiVersion expected=$PixiVersion"
}

$lockHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $ProjectRoot "pixi.lock")).Hash.ToLowerInvariant()
if ($lockHash -ne $LockSha256) {
    throw "pixi.lock hash mismatch: actual=$lockHash expected=$LockSha256"
}

if (Test-Path $ProjectPixi) {
    $projectPixiItem = Get-Item -Force $ProjectPixi
    $resolvedRuntimePixi = (Resolve-Path $PixiRoot).Path.TrimEnd("\")
    if (-not $projectPixiItem.LinkType) {
        throw "Repository .pixi must be a Junction to the D: runtime"
    }
    $junctionTarget = [System.IO.Path]::GetFullPath([string]$projectPixiItem.Target).TrimEnd("\")
    if ($junctionTarget -ne $resolvedRuntimePixi) {
        throw "Repository .pixi points to the wrong runtime: actual=$junctionTarget expected=$resolvedRuntimePixi"
    }
} elseif ($CheckOnly) {
    throw "Repository .pixi Junction not found: $ProjectPixi"
} else {
    New-Item -ItemType Junction -Path $ProjectPixi -Target $PixiRoot | Out-Null
}

if (-not $CheckOnly -and -not (Test-Path (Join-Path $IsaacLabRoot ".git"))) {
    Invoke-Checked -FilePath gh -ArgumentList @(
        "repo", "clone", "isaac-sim/IsaacLab", $IsaacLabRoot, "--", "--filter=blob:none"
    )
}
if (-not (Test-Path (Join-Path $IsaacLabRoot ".git"))) {
    throw "IsaacLab source checkout not found: $IsaacLabRoot"
}

if (-not $CheckOnly) {
    $dirty = & git -C $IsaacLabRoot status --porcelain
    if ($dirty) {
        throw "IsaacLab checkout has uncommitted changes"
    }
    & git -C $IsaacLabRoot cat-file -e "$IsaacLabCommit`^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Invoke-Checked -FilePath git -ArgumentList @(
            "-C", $IsaacLabRoot, "fetch", "--filter=blob:none", "origin", $IsaacLabCommit
        )
    }
    Invoke-Checked -FilePath git -ArgumentList @(
        "-C", $IsaacLabRoot, "checkout", "--detach", $IsaacLabCommit
    )
}

$actualIsaacLabCommit = (& git -C $IsaacLabRoot rev-parse HEAD).Trim()
if ($actualIsaacLabCommit -ne $IsaacLabCommit) {
    throw "IsaacLab commit mismatch: actual=$actualIsaacLabCommit expected=$IsaacLabCommit"
}

Invoke-Checked -FilePath $Pixi -ArgumentList @("lock", "--check")
if (-not $CheckOnly) {
    Invoke-Checked -FilePath $Pixi -ArgumentList @("install", "-e", "sim", "--locked")
    Invoke-Checked -FilePath $Pixi -ArgumentList @("install", "-e", "real", "--locked")
    Invoke-Checked -FilePath $Pixi -ArgumentList @("install", "-e", "ros-tools", "--locked")
}

Invoke-Checked -FilePath $Pixi -ArgumentList @("run", "stack-check-sim")
Invoke-Checked -FilePath $Pixi -ArgumentList @("run", "stack-check-real")
Invoke-Checked -FilePath $Pixi -ArgumentList @("run", "stack-check-ros")
Invoke-Checked -FilePath $Pixi -ArgumentList @("run", "ros-build")
Invoke-Checked -FilePath $Pixi -ArgumentList @("run", "-e", "ros-tools", "core-test")
Invoke-Checked -FilePath $Pixi -ArgumentList @("run", "-e", "real", "dataset-test")
Invoke-Checked -FilePath $Pixi -ArgumentList @(
    "run", "-e", "ros-tools", "python", "scripts/parity/validate_checkpoint.py",
    "--manifest", "configs/parity/runtime_manifest.mock.json",
    "--checkpoint", "configs/parity/replay_checkpoint.json"
)

$manifestLockHash = (& jq -r ".pixi_lock_hash" configs/parity/runtime_manifest.mock.json).Trim()
if ($manifestLockHash -ne $LockSha256) {
    throw "Runtime manifest pixi_lock_hash mismatch: $manifestLockHash"
}

if ($Smoke) {
    Invoke-Checked -FilePath $Pixi -ArgumentList @(
        "run", "sim-compatibility-check-headless",
        "--report", "outputs/parity/isaac_compatibility_windows.json"
    )
    Invoke-Checked -FilePath $Pixi -ArgumentList @(
        "run", "-e", "sim", "python", "scripts/parity/isaac6_smoke.py",
        "--stage", "camera",
        "--steps", "5",
        "--report", "outputs/parity/isaac6_camera_smoke_windows.json",
        "--visualizer", "none"
    )
}

if ($Probe) {
    $env:ZENOH_SESSION_CONFIG_URI = (Resolve-Path "configs/zenoh/windows-client.json5").Path
    Invoke-Checked -FilePath $Pixi -ArgumentList @(
        "run", "-e", "ros-tools", "python", "-m", "so101_vla_runtime.integration_probe",
        "--samples", "100",
        "--warmup", "5",
        "--image-pattern", "gradient",
        "--report", "outputs/parity/zenoh_probe_windows.json"
    )
}

$manifestHash = (& jq -r ".manifest_hash" configs/parity/runtime_manifest.mock.json).Trim()
Write-Output "status=passed"
Write-Output "runtime_root=$RuntimeRoot"
Write-Output "pixi_version=$actualPixiVersion"
Write-Output "pixi_lock_sha256=$lockHash"
Write-Output "isaaclab_commit=$actualIsaacLabCommit"
Write-Output "runtime_manifest_hash=$manifestHash"
Write-Output "smoke_executed=$($Smoke.IsPresent)"
Write-Output "transport_probe_executed=$($Probe.IsPresent)"
