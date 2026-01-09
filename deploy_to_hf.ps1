# Hugging Face Deployment Helper Script
# This script helps prepare your project for Hugging Face Spaces deployment

Write-Host "🚀 Hugging Face Spaces Deployment Helper" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Get project path
$projectPath = "d:\Y5 AMS\Sentiment_Analysis_of_Khmer_Text_Using_ML"

# Ask for Space directory
Write-Host "Please enter the path to your cloned Hugging Face Space directory:" -ForegroundColor Yellow
Write-Host "(e.g., C:\Users\YourName\khmer-sentiment-analysis)" -ForegroundColor Gray
$spacePath = Read-Host "Space path"

if (-not (Test-Path $spacePath)) {
    Write-Host "❌ Error: Space directory not found!" -ForegroundColor Red
    Write-Host "Please clone your Space first:" -ForegroundColor Yellow
    Write-Host "git clone https://huggingface.co/spaces/YOUR_USERNAME/khmer-sentiment-analysis" -ForegroundColor Gray
    exit 1
}

Write-Host "`n📦 Copying files..." -ForegroundColor Cyan

# Copy main app file (rename to app.py)
Write-Host "  ✓ Copying app_gradio.py -> app.py" -ForegroundColor Green
Copy-Item "$projectPath\app_gradio.py" -Destination "$spacePath\app.py" -Force

# Copy requirements
Write-Host "  ✓ Copying requirements_hf.txt -> requirements.txt" -ForegroundColor Green
Copy-Item "$projectPath\requirements_hf.txt" -Destination "$spacePath\requirements.txt" -Force

# Copy README
Write-Host "  ✓ Copying README_HF.md -> README.md" -ForegroundColor Green
Copy-Item "$projectPath\README_HF.md" -Destination "$spacePath\README.md" -Force

# Copy .gitattributes
Write-Host "  ✓ Copying .gitattributes" -ForegroundColor Green
Copy-Item "$projectPath\.gitattributes" -Destination "$spacePath\.gitattributes" -Force

# Copy source code
Write-Host "  ✓ Copying src/ directory" -ForegroundColor Green
if (Test-Path "$spacePath\src") {
    Remove-Item "$spacePath\src" -Recurse -Force
}
Copy-Item "$projectPath\src" -Destination "$spacePath\src" -Recurse -Force

# Copy model files
Write-Host "  ✓ Copying model files" -ForegroundColor Green
New-Item -ItemType Directory -Force -Path "$spacePath\models\saved_models" | Out-Null

# Check for models in different locations
$modelsFound = $false

if (Test-Path "$projectPath\models\saved_models\best_model_*.pkl") {
    Copy-Item "$projectPath\models\saved_models\best_model_*.pkl" -Destination "$spacePath\models\saved_models\" -Force
    Copy-Item "$projectPath\models\saved_models\best_model_*_metadata.json" -Destination "$spacePath\models\saved_models\" -Force -ErrorAction SilentlyContinue
    $modelsFound = $true
    Write-Host "    Found models in models/saved_models/" -ForegroundColor Gray
}

if (Test-Path "$projectPath\notebooks\best_model_*.joblib") {
    Copy-Item "$projectPath\notebooks\best_model_*.joblib" -Destination "$spacePath\" -Force
    $modelsFound = $true
    Write-Host "    Found models in notebooks/" -ForegroundColor Gray
}

if (-not $modelsFound) {
    Write-Host "  ⚠️  Warning: No model files found!" -ForegroundColor Yellow
    Write-Host "    Make sure you have trained models in:" -ForegroundColor Yellow
    Write-Host "    - models/saved_models/best_model_*.pkl" -ForegroundColor Gray
    Write-Host "    - notebooks/best_model_*.joblib" -ForegroundColor Gray
}

Write-Host "`n✅ Files copied successfully!" -ForegroundColor Green

Write-Host "`n📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Navigate to your Space directory:" -ForegroundColor White
Write-Host "   cd `"$spacePath`"" -ForegroundColor Gray

Write-Host "`n2. Initialize Git LFS (if not done already):" -ForegroundColor White
Write-Host "   git lfs install" -ForegroundColor Gray

Write-Host "`n3. Track model files with Git LFS:" -ForegroundColor White
Write-Host "   git lfs track `"*.pkl`"" -ForegroundColor Gray
Write-Host "   git lfs track `"*.joblib`"" -ForegroundColor Gray

Write-Host "`n4. Add all files:" -ForegroundColor White
Write-Host "   git add ." -ForegroundColor Gray

Write-Host "`n5. Commit changes:" -ForegroundColor White
Write-Host "   git commit -m `"Initial deployment: Khmer Sentiment Analysis`"" -ForegroundColor Gray

Write-Host "`n6. Push to Hugging Face:" -ForegroundColor White
Write-Host "   git push" -ForegroundColor Gray

Write-Host "`n📖 For detailed instructions, see:" -ForegroundColor Cyan
Write-Host "   $projectPath\DEPLOYMENT_GUIDE.md" -ForegroundColor Gray

Write-Host "`n🎉 Ready to deploy!" -ForegroundColor Green
