"""
Setup file for Khmer Sentiment Analysis package
"""

from setuptools import setup, find_packages

with open("README_PROFESSIONAL.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="khmer-sentiment-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Professional Khmer sentiment analysis system using ML and DL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/khmer-sentiment-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
        ],
        "deep_learning": [
            "tensorflow>=2.10.0",
            "keras>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "khmer-train=train:main",
            "khmer-predict=predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["*.json", "*.txt"],
    },
)
