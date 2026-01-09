#!/usr/bin/env python3
"""
Hugging Face Deployment Helper Script (Python Version)
This script helps prepare your project for Hugging Face Spaces deployment
"""

import os
import shutil
from pathlib import Path
import glob

def print_colored(text, color='cyan'):
    """Print colored text"""
    colors = {
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'gray': '\033[90m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def main():
    print_colored("🚀 Hugging Face Spaces Deployment Helper", 'cyan')
    print_colored("=" * 40 + "\n", 'cyan')
    
    # Get project path
    project_path = Path(r"D:\Y5 AMS\Sentiment-Analysis-of-Khmer-Text-Using-ML")
    
    # Ask for Space directory
    print_colored("Please enter the path to your cloned Hugging Face Space directory:", 'yellow')
    print_colored("(e.g., C:\\Users\\YourName\\khmer-sentiment-analysis)", 'gray')
    space_path_str = input("Space path: ").strip().strip('"')
    space_path = Path(space_path_str)
    
    if not space_path.exists():
        print_colored("\n❌ Error: Space directory not found!", 'red')
        print_colored("Please clone your Space first:", 'yellow')
        print_colored("git clone https://huggingface.co/spaces/YOUR_USERNAME/khmer-sentiment-analysis", 'gray')
        return 1
    
    print_colored("\n📦 Copying files...", 'cyan')
    
    try:
        # Copy main app file (rename to app.py)
        print_colored("  ✓ Copying app_gradio.py -> app.py", 'green')
        shutil.copy2(project_path / "app_gradio.py", space_path / "app.py")
        
        # Copy requirements
        print_colored("  ✓ Copying requirements_hf.txt -> requirements.txt", 'green')
        shutil.copy2(project_path / "requirements_hf.txt", space_path / "requirements.txt")
        
        # Copy README
        print_colored("  ✓ Copying README_HF.md -> README.md", 'green')
        shutil.copy2(project_path / "README_HF.md", space_path / "README.md")
        
        # Copy .gitattributes
        print_colored("  ✓ Copying .gitattributes", 'green')
        shutil.copy2(project_path / ".gitattributes", space_path / ".gitattributes")
        
        # Copy source code
        print_colored("  ✓ Copying src/ directory", 'green')
        if (space_path / "src").exists():
            shutil.rmtree(space_path / "src")
        shutil.copytree(project_path / "src", space_path / "src")
        
        # Copy model files
        print_colored("  ✓ Copying model files", 'green')
        (space_path / "models" / "saved_models").mkdir(parents=True, exist_ok=True)
        
        models_found = False
        
        # Check for models in models/saved_models/
        pkl_models = list((project_path / "models" / "saved_models").glob("best_model_*.pkl"))
        if pkl_models:
            for model in pkl_models:
                shutil.copy2(model, space_path / "models" / "saved_models")
                print_colored(f"    Copied {model.name}", 'gray')
            
            # Copy metadata files
            json_files = list((project_path / "models" / "saved_models").glob("best_model_*_metadata.json"))
            for json_file in json_files:
                shutil.copy2(json_file, space_path / "models" / "saved_models")
            
            models_found = True
            print_colored("    Found models in models/saved_models/", 'gray')
        
        # Check for models in notebooks/
        joblib_models = list((project_path / "notebooks").glob("best_model_*.joblib"))
        if joblib_models:
            for model in joblib_models:
                shutil.copy2(model, space_path)
                print_colored(f"    Copied {model.name}", 'gray')
            models_found = True
            print_colored("    Found models in notebooks/", 'gray')
        
        if not models_found:
            print_colored("  ⚠️  Warning: No model files found!", 'yellow')
            print_colored("    Make sure you have trained models in:", 'yellow')
            print_colored("    - models/saved_models/best_model_*.pkl", 'gray')
            print_colored("    - notebooks/best_model_*.joblib", 'gray')
        
        print_colored("\n✅ Files copied successfully!", 'green')
        
        # Print next steps
        print_colored("\n📋 Next steps:", 'cyan')
        print_colored("1. Navigate to your Space directory:", 'white')
        print_colored(f'   cd "{space_path}"', 'gray')
        
        print_colored("\n2. Initialize Git LFS (if not done already):", 'white')
        print_colored("   git lfs install", 'gray')
        
        print_colored("\n3. Track model files with Git LFS:", 'white')
        print_colored('   git lfs track "*.pkl"', 'gray')
        print_colored('   git lfs track "*.joblib"', 'gray')
        
        print_colored("\n4. Add all files:", 'white')
        print_colored("   git add .", 'gray')
        
        print_colored("\n5. Commit changes:", 'white')
        print_colored('   git commit -m "Initial deployment: Khmer Sentiment Analysis"', 'gray')
        
        print_colored("\n6. Push to Hugging Face:", 'white')
        print_colored("   git push", 'gray')
        
        print_colored("\n📖 For detailed instructions, see:", 'cyan')
        print_colored(f"   {project_path / 'DEPLOYMENT_GUIDE.md'}", 'gray')
        
        print_colored("\n🎉 Ready to deploy!", 'green')
        
        return 0
        
    except Exception as e:
        print_colored(f"\n❌ Error: {str(e)}", 'red')
        return 1

if __name__ == "__main__":
    exit(main())
