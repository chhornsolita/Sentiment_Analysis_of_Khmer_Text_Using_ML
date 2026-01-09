# 🚀 Deployment Guide: Hugging Face Spaces

This guide will walk you through deploying your Khmer Sentiment Analysis project to Hugging Face Spaces.

## 📋 Prerequisites

1. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co/join)
2. **Git**: Installed on your local machine
3. **Git LFS**: Install Git Large File Storage for model files

## 🔧 Setup Steps

### Step 1: Install Git LFS

Git LFS is required for uploading model files (_.pkl, _.joblib):

**Windows:**

```powershell
# Download from https://git-lfs.github.com/
# Or use chocolatey
choco install git-lfs

# Initialize Git LFS
git lfs install
```

**Linux/Mac:**

```bash
# Install Git LFS
sudo apt-get install git-lfs  # Ubuntu/Debian
brew install git-lfs           # Mac

# Initialize Git LFS
git lfs install
```

### Step 2: Create a New Space on Hugging Face

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the details:

   - **Space name**: `khmer-sentiment-analysis` (or your choice)
   - **License**: MIT
   - **Select SDK**: Gradio
   - **Space hardware**: CPU Basic (free tier)
   - **Visibility**: Public or Private

4. Click **"Create Space"**

### Step 3: Clone the Space Repository

```bash
# Replace YOUR_USERNAME with your Hugging Face username
git clone https://huggingface.co/spaces/YOUR_USERNAME/khmer-sentiment-analysis
cd khmer-sentiment-analysis
```

### Step 4: Prepare Files for Upload

Copy these files from your project to the cloned Space directory:

```bash
# Navigate to your project directory
cd "d:\Y5 AMS\Sentiment-Analysis-of-Khmer-Text-Using-ML"

# Copy essential files to the Space directory
# (Adjust the paths according to your Space clone location)

# Copy the main app file (rename to app.py for Hugging Face)
cp app_gradio.py /path/to/khmer-sentiment-analysis/app.py

# Copy requirements
cp requirements_hf.txt /path/to/khmer-sentiment-analysis/requirements.txt

# Copy README
cp README_HF.md /path/to/khmer-sentiment-analysis/README.md

# Copy .gitattributes
cp .gitattributes /path/to/khmer-sentiment-analysis/.gitattributes

# Copy source code folder
cp -r src /path/to/khmer-sentiment-analysis/

# Copy model files (choose the best model)
mkdir -p /path/to/khmer-sentiment-analysis/models/saved_models
cp models/saved_models/best_model_*.pkl /path/to/khmer-sentiment-analysis/models/saved_models/
# Or copy from notebooks if using joblib
cp notebooks/best_model_*.joblib /path/to/khmer-sentiment-analysis/

# Copy necessary data files if needed by preprocessing
mkdir -p /path/to/khmer-sentiment-analysis/data
# Only copy if needed for slang dictionary or other preprocessing resources
```

**PowerShell Commands (Windows):**

```powershell
# Set variables for paths
$projectPath = "d:\Y5 AMS\Sentiment-Analysis-of-Khmer-Text-Using-ML"
$spacePath = "path\to\khmer-sentiment-analysis"

# Copy files
Copy-Item "$projectPath\app_gradio.py" -Destination "$spacePath\app.py"
Copy-Item "$projectPath\requirements_hf.txt" -Destination "$spacePath\requirements.txt"
Copy-Item "$projectPath\README_HF.md" -Destination "$spacePath\README.md"
Copy-Item "$projectPath\.gitattributes" -Destination "$spacePath\.gitattributes"

# Copy directories
Copy-Item "$projectPath\src" -Destination "$spacePath\src" -Recurse

# Copy model files
New-Item -ItemType Directory -Force -Path "$spacePath\models\saved_models"
Copy-Item "$projectPath\models\saved_models\best_model_*.pkl" -Destination "$spacePath\models\saved_models\"
# Or from notebooks
Copy-Item "$projectPath\notebooks\best_model_*.joblib" -Destination "$spacePath\"
```

### Step 5: Verify File Structure

Your Space directory should look like this:

```
khmer-sentiment-analysis/
├── .gitattributes           # Git LFS configuration
├── README.md                # With HF metadata header
├── app.py                   # Main Gradio app (renamed from app_gradio.py)
├── requirements.txt         # Python dependencies
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model_persistence.py
│   └── ...
└── models/
    └── saved_models/
        ├── best_model_*.pkl  # Model file
        └── best_model_*_metadata.json  # Model metadata (if exists)
```

### Step 6: Commit and Push to Hugging Face

```bash
cd /path/to/khmer-sentiment-analysis

# Track model files with Git LFS
git lfs track "*.pkl"
git lfs track "*.joblib"

# Add all files
git add .

# Commit changes
git commit -m "Initial deployment: Khmer Sentiment Analysis app"

# Push to Hugging Face
git push
```

**Note**: The first push might take some time as it uploads the model files via Git LFS.

### Step 7: Wait for Build

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/khmer-sentiment-analysis`
2. Watch the build logs in the "Settings" > "Factory reboot" section
3. The Space will automatically build and start once the upload is complete
4. First build typically takes 2-5 minutes

### Step 8: Test Your App

Once deployed, your app will be available at:

```
https://YOUR_USERNAME-khmer-sentiment-analysis.hf.space
```

Try the example texts to verify everything works correctly!

## 🔍 Troubleshooting

### Issue: Model File Too Large

If your model file is larger than 5GB, you'll need to:

1. Use Git LFS (already configured)
2. Or optimize your model size
3. Or consider using Hugging Face's model hub

### Issue: Import Errors

If you see import errors in the build logs:

1. Check that all required packages are in `requirements.txt`
2. Ensure version compatibility
3. Verify that all imports in `app.py` are correct

### Issue: App Not Loading

If the app builds but doesn't load:

1. Check the logs in the Space's "Settings" tab
2. Verify model files are in the correct location
3. Test locally first with: `python app.py`

### Issue: Git LFS Quota Exceeded

Free tier has 10GB storage:

1. Only upload necessary model files
2. Consider upgrading to Pro ($9/month) for 100GB
3. Or compress/optimize models

## 🎨 Customization

### Update App Title and Description

Edit the header in `README.md`:

```yaml
---
title: Your Custom Title
emoji: 🎯 # Choose an emoji
colorFrom: blue
colorTo: green
---
```

### Change Theme

In `app.py`, modify the theme:

```python
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Change to: gr.themes.Default(), gr.themes.Glass(), etc.
```

### Add Custom CSS

Add custom styling in the `css` parameter:

```python
with gr.Blocks(
    theme=gr.themes.Soft(),
    css="your-custom-css-here"
) as demo:
```

## 📊 Monitor Usage

1. Go to your Space settings
2. View analytics under "Settings" > "Analytics"
3. Monitor:
   - Number of visits
   - API calls
   - Build status

## 🔄 Updating Your Space

To update your deployed app:

```bash
cd /path/to/khmer-sentiment-analysis

# Make changes to files
# Then commit and push
git add .
git commit -m "Update: description of changes"
git push

# The Space will automatically rebuild
```

## 💰 Upgrade Options

**Free Tier (CPU Basic):**

- 2 vCPU, 16GB RAM
- Suitable for most use cases
- May have slower response times under load

**Upgraded Hardware:**

- CPU Upgrade: $0.60/hour (better performance)
- GPU: For deep learning models (if you add LSTM/Transformer)
- Visit Space Settings > Hardware to upgrade

## 🔗 Share Your Space

Once deployed, share your Space:

1. **Direct Link**: `https://YOUR_USERNAME-khmer-sentiment-analysis.hf.space`
2. **Embed**: Use the embed code in your website
3. **API**: Use Gradio's automatic API endpoint

### API Usage Example:

```python
from gradio_client import Client

client = Client("YOUR_USERNAME/khmer-sentiment-analysis")
result = client.predict("ល្អណាស់!", api_name="/predict")
print(result)
```

## ✅ Post-Deployment Checklist

- [ ] Space builds successfully without errors
- [ ] App interface loads correctly
- [ ] Test predictions work with Khmer text
- [ ] Examples work as expected
- [ ] README displays properly with metadata
- [ ] Share link with others for testing

## 🆘 Need Help?

- **Hugging Face Docs**: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Gradio Docs**: [gradio.app/docs](https://gradio.app/docs)
- **Community**: Hugging Face Discord and Forums

---

**Congratulations! Your Khmer Sentiment Analysis app is now live on Hugging Face Spaces! 🎉**
