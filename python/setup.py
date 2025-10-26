from setuptools import setup, find_packages

setup(
    name="flashcompile",
    version="0.1.0",
    description="Python API for Flash ML Compiler",
    author="Jay Chawrey",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    python_requires=">=3.8",
)