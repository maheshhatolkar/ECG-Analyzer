#!/usr/bin/env pwsh
param(
    [string]$entry = 'run_gui.py',
    [string]$name = 'ECGAnalyzer',
    [switch]$onefile
)

Write-Host "Building desktop app with PyInstaller..."

# Ensure pyinstaller is installed in the active env: pip install pyinstaller
$args = @()
if ($onefile) { $args += '--onefile' }
$args += '--name'; $args += $name
$args += '--noconfirm'; $args += '--windowed'

# include templates and web assets if packing the web UI later
$spec_data = "--add-data `"web/templates;web/templates`""
$args += $spec_data
$args += $entry

pyinstaller @args

Write-Host "Build finished. See dist\$name for artifacts."
