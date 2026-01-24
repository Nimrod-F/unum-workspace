#!/usr/bin/env pwsh
# Unum CLI Wrapper for PowerShell
# This script makes it easier to run unum-cli commands

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$unumCliPath = Join-Path $scriptDir "unum\unum-cli\unum-cli.py"

py $unumCliPath $args
