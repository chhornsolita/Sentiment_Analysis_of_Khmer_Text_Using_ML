# 📋 Hugging Face Deployment Checklist

Use this checklist to track your deployment progress.

## Pre-Deployment

- [ ] Hugging Face account created
- [ ] Git installed and configured
- [ ] Git LFS installed (`git lfs install`)
- [ ] Model files are trained and ready
- [ ] Tested app locally (optional but recommended)

## Create Space

- [ ] Visited https://huggingface.co/spaces
- [ ] Clicked "Create new Space"
- [ ] Entered Space name: `khmer-sentiment-analysis`
- [ ] Selected SDK: **Gradio**
- [ ] Selected License: MIT
- [ ] Set visibility: Public or Private
- [ ] Clicked "Create Space"
- [ ] Noted Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/khmer-sentiment-analysis`

## Prepare Local Files

- [ ] Cloned Space repository locally
- [ ] Ran deployment script: `.\deploy_to_hf.ps1` OR manually copied files:
  - [ ] Copied `app_gradio.py` → `app.py`
  - [ ] Copied `requirements_hf.txt` → `requirements.txt`
  - [ ] Copied `README_HF.md` → `README.md`
  - [ ] Copied `.gitattributes`
  - [ ] Copied `src/` folder
  - [ ] Copied model files to `models/saved_models/`
- [ ] Verified all files are in the Space directory

## Git Configuration

- [ ] Navigated to Space directory: `cd khmer-sentiment-analysis`
- [ ] Initialized Git LFS: `git lfs install`
- [ ] Tracked model files:
  - [ ] `git lfs track "*.pkl"`
  - [ ] `git lfs track "*.joblib"`
- [ ] Verified LFS tracking: `git lfs track`

## Commit and Push

- [ ] Added files: `git add .`
- [ ] Checked status: `git status`
- [ ] Committed changes: `git commit -m "Initial deployment: Khmer Sentiment Analysis"`
- [ ] Pushed to Hugging Face: `git push`
- [ ] Confirmed push completed successfully (no errors)

## Verify Deployment

- [ ] Visited Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/khmer-sentiment-analysis`
- [ ] Watched build logs (Settings tab if needed)
- [ ] Build completed successfully (green checkmark)
- [ ] Space status shows "Running"

## Test Application

- [ ] App interface loads correctly
- [ ] Tested with positive example: "ល្អណាស់! វាពិតជាអស្ចារ្យ"
- [ ] Tested with negative example: "គួរឱ្យខកចិត្ត មិនល្អទេ"
- [ ] Tested with neutral example: "ធម្មតាៗ មិនអីទេ"
- [ ] Confidence scores display correctly
- [ ] Examples buttons work
- [ ] No error messages in the interface

## Post-Deployment

- [ ] README displays correctly with metadata
- [ ] Updated personal README or portfolio with Space link
- [ ] Shared Space with friends/colleagues for testing
- [ ] Noted feedback for future improvements

## Optional Enhancements

- [ ] Customized Space appearance (theme, colors)
- [ ] Added custom examples relevant to your use case
- [ ] Set up Space analytics tracking
- [ ] Considered upgrading hardware if needed
- [ ] Created API documentation for Space
- [ ] Added Space badge to GitHub README

## Troubleshooting (if needed)

- [ ] Checked build logs for errors
- [ ] Verified all dependencies in requirements.txt
- [ ] Confirmed model files uploaded via Git LFS
- [ ] Tested imports locally
- [ ] Checked file paths are correct
- [ ] Reviewed DEPLOYMENT_GUIDE.md for solutions

## Maintenance

- [ ] Noted Space URL for future reference
- [ ] Set up process for updating model
- [ ] Documented how to push updates
- [ ] Created monitoring plan for usage/errors

---

## Notes & Issues

_Use this space to track any issues or notes during deployment:_

**Issue 1:**

- Description:
- Solution:

**Issue 2:**

- Description:
- Solution:

**Deployment Date:** ******\_\_\_******

**Space URL:** **********************\_\_\_**********************

**Status:** ⬜ Not Started | ⬜ In Progress | ⬜ Completed | ⬜ Issues

---

**🎉 Congratulations on deploying your Khmer Sentiment Analysis app!**
