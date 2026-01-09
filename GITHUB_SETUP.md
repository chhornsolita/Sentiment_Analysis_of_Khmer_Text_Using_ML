# GitHub Repository Setup Guide

Your local Git repository has been initialized and committed. Now push it to GitHub:

## Quick Setup (Copy-paste these commands)

```powershell
# Replace YOUR_USERNAME with your actual GitHub username
cd "D:\Y5 AMS\Sentiment_Analysis_of_Khmer_Text_Using_ML"

# Add GitHub remote (change YOUR_USERNAME!)
git remote add origin https://github.com/YOUR_USERNAME/Sentiment_Analysis_of_Khmer_Text_Using_ML.git

# Rename branch to main (GitHub's default)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Detailed Steps

### 1. Create GitHub Repository

- Go to: https://github.com/new
- Repository name: `Sentiment_Analysis_of_Khmer_Text_Using_ML`
- **Important**: Leave all checkboxes UNCHECKED (no README, .gitignore, or license)
- Click "Create repository"

### 2. Copy Your Repository URL

After creating, you'll see a URL like:

```
https://github.com/YOUR_USERNAME/Sentiment_Analysis_of_Khmer_Text_Using_ML.git
```

### 3. Run the Commands Above

Replace `YOUR_USERNAME` with your actual GitHub username in the commands.

## Verify Upload

After pushing, visit:

```
https://github.com/YOUR_USERNAME/Sentiment_Analysis_of_Khmer_Text_Using_ML
```

You should see all 79 files uploaded!

## Troubleshooting

### Authentication Required

If prompted for credentials:

- Username: Your GitHub username
- Password: Use a **Personal Access Token** (not your password)
  - Create token at: https://github.com/settings/tokens
  - Select scope: `repo` (full control of private repositories)

### Already Exists Error

If you get "repository already exists":

```powershell
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/CORRECT_REPO_NAME.git
git push -u origin main
```

## What's Been Done

✅ Initialized Git repository
✅ Added all 79 files
✅ Created initial commit
✅ Fixed nested repository issue (khmer-sentiment-analysis)

## Next: Push to GitHub

Run the commands at the top of this file!
