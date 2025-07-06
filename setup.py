from setuptools import setup, find_packages

setup(
    name="ttf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "pyyaml",
    ],
    python_requires=">=3.8",
) 