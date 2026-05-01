from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="overcooked-with-AI",
    version="1.0.2",
    author="ZSC-Eval Team",
    description="Overcooked Human-AI Interaction Testing Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ZSC-Eval",
    packages=find_packages(),
    # experiment_manager and crypto_utils are now part of the human_test package
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0,<2.0.0",
        "gym==0.22.0",
        "scipy>=1.7.0",
        "pygame>=2.5.0",
        "pyyaml>=6.0",
        "colorama>=0.4.0",
        "pynput>=1.7.0",
        "pyfiglet>=0.8.0",
        "cryptography>=41.0.0",
        "tqdm>=4.60.0",
        "loguru>=0.7.0",
        "absl-py>=1.0.0",
        "ipython>=8.0.0",
        "ipywidgets>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "overcooked-human-test=human_test.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
