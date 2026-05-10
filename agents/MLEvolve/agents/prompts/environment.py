"""Environment/package prompt."""

import random


def get_prompt_environment():
    """Installed packages description."""
    pkgs = [
        "numpy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "xgboost",
        "lightGBM",
        "torch",
        "torchvision",
        "torch-geometric",
        "bayesian-optimization",
        "timm",
        "transformers",
        "sentence-transformers",
        "opencv-python",
        "Pillow",
    ]
    random.shuffle(pkgs)
    pkg_str = ", ".join([f"`{p}`" for p in pkgs])

    return {
        "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
    }
