# Quick Start Script for ML API
# Run this script to set up and start the Flask API

Write-Host "=== Benguet Crop ML API Setup ===" -ForegroundColor Green
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Check if model files exist
Write-Host ""
Write-Host "Checking model files..." -ForegroundColor Yellow
$requiredFiles = @(
    "model_artifacts\best_model.pkl",
    "model_artifacts\best_rf_model.pkl",
    "model_artifacts\model_metadata.json",
    "model_artifacts\feature_statistics.json"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  [OK] $file" -ForegroundColor Green
    }
    else {
        Write-Host "  [MISSING] $file" -ForegroundColor Red
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "ERROR: Missing required model files!" -ForegroundColor Red
    Write-Host "Please ensure you have trained the model first using the Jupyter notebook." -ForegroundColor Yellow
    exit 1
}

# Install dependencies
Write-Host ""
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Starting Flask API on http://127.0.0.1:5000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "API Endpoints:" -ForegroundColor White
Write-Host "  - Home: http://127.0.0.1:5000/" -ForegroundColor Gray
Write-Host "  - Predict: POST http://127.0.0.1:5000/predict" -ForegroundColor Gray
Write-Host "  - Batch Predict: POST http://127.0.0.1:5000/batch-predict" -ForegroundColor Gray
Write-Host "  - Model Info: http://127.0.0.1:5000/model-info" -ForegroundColor Gray
Write-Host ""

# Start the Flask app
python ml_api.py
