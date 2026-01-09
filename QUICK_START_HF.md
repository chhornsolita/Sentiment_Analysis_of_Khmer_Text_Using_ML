# 🚀 Quick Start: Deploy to Hugging Face

## Prerequisites

- Hugging Face account ([sign up here](https://huggingface.co/join))
- Git installed
- Git LFS installed

## Quick Deploy Steps

### 1️⃣ Install Git LFS

```bash
git lfs install
```

### 2️⃣ Create New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Name: `khmer-sentiment-analysis`
4. SDK: **Gradio**
5. Click "Create Space"

### 3️⃣ Clone Your Space

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/khmer-sentiment-analysis
```

### 4️⃣ Copy Files (Automated)

**Option A - Use PowerShell Script:**

```powershell
.\deploy_to_hf.ps1
```

**Option B - Manual Copy:**

```bash
# Copy to your cloned space directory:
- app_gradio.py → app.py
- requirements_hf.txt → requirements.txt
- README_HF.md → README.md
- .gitattributes
- src/ (entire folder)
- models/saved_models/ (model files)
```

### 5️⃣ Push to Hugging Face

```bash
cd khmer-sentiment-analysis
git lfs track "*.pkl"
git lfs track "*.joblib"
git add .
git commit -m "Initial deployment"
git push
```

### 6️⃣ Access Your App

Your app will be live at:

```
https://YOUR_USERNAME-khmer-sentiment-analysis.hf.space
```

## 📝 Files Created for Deployment

| File                  | Purpose                                       |
| --------------------- | --------------------------------------------- |
| `app_gradio.py`       | Gradio interface (rename to app.py for HF)    |
| `requirements_hf.txt` | Python dependencies for Hugging Face          |
| `README_HF.md`        | README with HF metadata (rename to README.md) |
| `.gitattributes`      | Git LFS configuration for model files         |
| `deploy_to_hf.ps1`    | PowerShell script to automate file copying    |
| `DEPLOYMENT_GUIDE.md` | Detailed deployment instructions              |

## ⚡ Troubleshooting

**Model not loading?**

- Check model files are in `models/saved_models/`
- Verify Git LFS tracked the files: `git lfs ls-files`

**Build failing?**

- Check requirements.txt has all dependencies
- View logs in Space settings

**App not responding?**

- First build takes 2-5 minutes
- Check Space logs for errors

## 📚 Documentation

- Full Guide: `DEPLOYMENT_GUIDE.md`
- Hugging Face Docs: https://huggingface.co/docs/hub/spaces
- Gradio Docs: https://gradio.app/docs

## 🆘 Need Help?

See the detailed `DEPLOYMENT_GUIDE.md` or visit Hugging Face community forums.
