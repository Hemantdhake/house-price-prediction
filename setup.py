from setuptools import setup, find_packages

setup(
    name="house_price_prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="House Price Prediction using Scikit-learn",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/house-price-prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.1",
        "scikit-learn>=1.2.2",
        "xgboost>=1.7.5",
        "flask>=2.3.2",
        "joblib>=1.2.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "train-model=src.model:main",
            "predict=src.predict:main",
        ],
    },
)