import os

from setuptools import find_packages, setup


# Read requirements from requirements.txt, filtering out git URLs
def read_requirements(filename):
    """Read requirements from a file, filtering out comments, empty lines, and git URLs."""
    requirements_file = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(requirements_file):
        return []
    
    with open(requirements_file, "r", encoding="utf-8") as fh:
        requirements = []
        for line in fh:
            line = line.strip()
            # Skip comments, empty lines, and git URLs
            if line and not line.startswith("#") and not line.startswith("git+"):
                requirements.append(line)
    return requirements

# Read requirements from requirements.txt (git URLs will be handled by pip during installation)
requirements = read_requirements("requirements.txt")

setup(
    name="VolAlign",
    version="0.1.0",
    packages=find_packages(),
    description="A comprehensive Python package for volumetric image alignment, stitching, and processing",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Mansour Alawi",
    author_email="mansour@xpress.ai",
    url="https://github.com/XpressAI/VolAlign",
    license="MIT",
    install_requires=requirements,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
            'black',
            'flake8',
            'mypy'
        ]
    }
)