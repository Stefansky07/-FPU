"""
Setup script for the Fused Private Update (FPU) package.
"""

from setuptools import setup, find_packages

setup(
    name="fused_private_update",
    version="0.1.0",
    description="GPU-accelerated fused kernel for privacy-preserving federated learning",
    author="Research Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "triton>=2.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
