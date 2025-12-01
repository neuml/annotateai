# pylint: disable = C0111
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    # Remove GitHub dark mode images
    DESCRIPTION = "".join([line for line in f if "gh-dark-mode-only" not in line])

setup(
    name="annotateai",
    version="0.4.0",
    author="NeuML",
    description="Automatically annotate papers using LLMs",
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/neuml/annotateai",
    project_urls={
        "Documentation": "https://github.com/neuml/annotateai",
        "Issue Tracker": "https://github.com/neuml/annotateai/issues",
        "Source Code": "https://github.com/neuml/annotateai",
    },
    license="Apache 2.0: http://www.apache.org/licenses/LICENSE-2.0",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    keywords="pdf highlight llm ai",
    python_requires=">=3.10",
    install_requires=["nltk>=3.5", "tqdm>=4.48.0", "txtai>=9.2.0", "txtmarker>=1.1.0"],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Utilities",
    ],
)
