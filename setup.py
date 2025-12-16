from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="camadrl",
    version="0.1.0",
    author="Mudds Air Sharif",
    author_email="muddsairsharif@example.com",
    description="CAMADRL: A package for multi-agent reinforcement learning research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/muddsairsharif/CAMADRL",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "gym>=0.18.0",
        "gymnasium>=0.26.0",
        "networkx>=2.5",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "tensorboard>=2.6.0",
        "pyyaml>=5.3.1",
        "h5py>=2.10.0",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
