<#
.SYNOPSIS
    Full Benchmark Runner for Progressive-Aggregator

.DESCRIPTION
    Runs comprehensive benchmarks for all three execution modes:
    - CLASSIC (traditional last-invoker)
    - EAGER (LazyInput polling)
    - FUTURE_BASED (async with background polling)

.PARAMETER Iterations
    Number of warm iterations per mode (default: 10)

.PARAMETER WarmupRuns
    Number of warmup runs before benchmark (default: 2)

.PARAMETER ColdRuns
    Number of cold start runs (default: 0)

.PARAMETER SkipDeploy
    Skip mode switching and deployment

.EXAMPLE
    .\run_full_benchmark.ps1 -Iterations 10
    .\run_full_benchmark.ps1 -Iterations 30 -ColdRuns 5
    .\run_full_benchmark.ps1 -SkipDeploy
#>

param(
    [int]$Iterations = 10,
    [int]$WarmupRuns = 2,
    [int]$ColdRuns = 0,
    [switch]$SkipDeploy
)

$ErrorActionPreference = "Stop"

# Paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$UnumCliPath = Join-Path (Split-Path -Parent (Split-Path -Parent $ProjectDir)) "unum\unum-cli\unum-cli.py"

Write-Host ""
Write-Host "=" * 60
Write-Host "  Progressive-Aggregator Full Benchmark"
Write-Host "=" * 60
Write-Host "  Iterations: $Iterations"
Write-Host "  Warmup: $WarmupRuns"
Write-Host "  Cold runs: $ColdRuns"
Write-Host "  Skip deploy: $SkipDeploy"
Write-Host ""

$Modes = @("CLASSIC", "EAGER", "FUTURE_BASED")
$Results = @{}

foreach ($Mode in $Modes) {
    Write-Host ""
    Write-Host "#" * 60
    Write-Host "  MODE: $Mode"
    Write-Host "#" * 60
    
    if (-not $SkipDeploy) {
        # Update unum-template.yaml for this mode
        Write-Host "  Updating configuration for $Mode mode..."
        
        $TemplateFile = Join-Path $ProjectDir "unum-template.yaml"
        $Content = Get-Content $TemplateFile -Raw
        
        switch ($Mode) {
            "CLASSIC" {
                $Content = $Content -replace "Eager:\s*(True|False)", "Eager: False"
                $Content = $Content -replace 'UNUM_FUTURE_BASED:\s*"?(true|false)"?', 'UNUM_FUTURE_BASED: "false"'
            }
            "EAGER" {
                $Content = $Content -replace "Eager:\s*(True|False)", "Eager: True"
                $Content = $Content -replace 'UNUM_FUTURE_BASED:\s*"?(true|false)"?', 'UNUM_FUTURE_BASED: "false"'
            }
            "FUTURE_BASED" {
                $Content = $Content -replace "Eager:\s*(True|False)", "Eager: True"
                $Content = $Content -replace 'UNUM_FUTURE_BASED:\s*"?(true|false)"?', 'UNUM_FUTURE_BASED: "true"'
            }
        }
        
        Set-Content -Path $TemplateFile -Value $Content
        
        # Rebuild and deploy
        Write-Host "  Rebuilding and deploying..."
        Push-Location $ProjectDir
        
        try {
            python $UnumCliPath template 2>&1 | Out-Null
            python $UnumCliPath build 2>&1 | Out-Null
            python $UnumCliPath deploy 2>&1 | Out-Null
            Write-Host "  Deploy successful!"
        }
        catch {
            Write-Host "  Deploy failed: $_"
            Pop-Location
            continue
        }
        
        Pop-Location
        
        # Wait for deployment to stabilize
        Write-Host "  Waiting for deployment to stabilize..."
        Start-Sleep -Seconds 15
    }
    
    # Run benchmark
    Write-Host "  Running benchmark..."
    
    $BenchmarkScript = Join-Path $ScriptDir "run_benchmark.py"
    $OutputDir = Join-Path $ScriptDir "results"
    
    $Args = @(
        "--mode", $Mode,
        "--iterations", $Iterations,
        "--warmup", $WarmupRuns,
        "--cold", $ColdRuns,
        "--output", $OutputDir,
        "--no-deploy"
    )
    
    python $BenchmarkScript @Args
    
    $Results[$Mode] = $true
}

Write-Host ""
Write-Host "=" * 60
Write-Host "  BENCHMARK COMPLETE"
Write-Host "=" * 60
Write-Host ""
Write-Host "  Results saved in: $OutputDir"
Write-Host ""
Write-Host "  Completed modes:"
foreach ($Mode in $Modes) {
    $Status = if ($Results[$Mode]) { "OK" } else { "FAILED" }
    Write-Host "    - $Mode : $Status"
}

Write-Host ""
Write-Host "  To analyze results:"
Write-Host "    python benchmark\analyze_results.py benchmark\results\"
Write-Host ""
Write-Host "  To generate charts:"
Write-Host "    python benchmark\generate_charts.py benchmark\results\ --output benchmark\figures\"
Write-Host ""
