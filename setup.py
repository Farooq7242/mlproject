from setuptools import setup, find_packages

setup(
    name="mlproject",
    version="0.1.0",
    author="Farooq Khan",
    author_email="farooqkhansec@gmail.com",
    description="A simple machine learning project setup",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "jupyter",
        "notebook",
        "tensorflow",
        "torch",
        "xgboost",
        "lightgbm",
        "joblib",
        "tqdm",
    ],
    python_requires=">=3.8",
)
