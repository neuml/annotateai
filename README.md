<p align="center">
    <img src="logo.png"/>
</p>

<p align="center">
    <b>Automatically annotate papers using LLMs</b>
</p>

<p align="center">
    <a href="https://github.com/neuml/annotate/releases">
        <img src="https://img.shields.io/github/release/neuml/annotateai.svg?style=flat&color=success" alt="Version"/>
    </a>
    <a href="https://github.com/neuml/annotate/releases">
        <img src="https://img.shields.io/github/release-date/neuml/annotateai.svg?style=flat&color=blue" alt="GitHub Release Date"/>
    </a>
    <a href="https://github.com/neuml/annotateai/issues">
        <img src="https://img.shields.io/github/issues/neuml/annotateai.svg?style=flat&color=success" alt="GitHub issues"/>
    </a>
    <a href="https://github.com/neuml/annotateai">
        <img src="https://img.shields.io/github/last-commit/neuml/annotateai.svg?style=flat&color=blue" alt="GitHub last commit"/>
    </a>
</p>

-------------------------------------------------------------------------------------------------------------------------------------------------------

![demo](https://raw.githubusercontent.com/neuml/annotateai/master/images/demo.png)

`annotateai` automatically annotates papers using Large Language Models (LLMs). While LLMs can summarize papers, search papers and build generative text about papers, this project focuses on providing human readers with context as they read.

## Architecture

![architecture](https://raw.githubusercontent.com/neuml/annotateai/master/images/architecture.png#gh-light-mode-only)
![architecture](https://raw.githubusercontent.com/neuml/annotateai/master/images/architecture-dark.png#gh-dark-mode-only)

A one-line call does the following:

- Reads the paper
- Finds the title and important key concepts
- Goes through each page and finds sections that best emphasis the key concepts
- Reads the section and builds a concise short topic
- Annotates the paper and highlights those sections

## Installation
The easiest way to install is via pip and PyPI

```
pip install annotateai
```

Python 3.9+ is supported. Using a Python [virtual environment](https://docs.python.org/3/library/venv.html) is recommended.

`annotateai` can also be installed directly from GitHub to access the latest, unreleased features.

```
pip install git+https://github.com/neuml/annotateai
```

Python 3.9+ is supported

## Examples

`annotateai` can annotate any PDF but it works especially well for medical and scientific papers. The following shows a series of examples using papers from [arXiv](https://arxiv.org/).

This project also works well with papers from [PubMed](https://pubmed.ncbi.nlm.nih.gov/), [bioRxiv](https://www.biorxiv.org/) and [medRxiv](https://www.medrxiv.org/)!

### Setup

The examples below use an AutoAWQ model, make sure to install that dependency first.

```
pip install annotateai autoawq[kernels]
```

The primary input parameter is the path to the LLM. This project is backed by [txtai](https://github.com/neuml/txtai) and it supports any [txtai-supported LLM](https://neuml.github.io/txtai/pipeline/text/llm/).

```python
from annotateai import Annotate

# This model works well with medical and scientific literature
annotate = Annotate("NeuML/Llama-3.1_OpenScholar-8B-AWQ")
```

### Annotate paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

This paper proposed RAG before most of us knew we needed it.

```python
annotate("https://arxiv.org/pdf/2005.11401")
```

![rag](https://raw.githubusercontent.com/neuml/annotateai/master/images/rag.png)

_Source: https://arxiv.org/pdf/2005.11401_

### Annotate paper "HunyuanVideo: A Systematic Framework For Large Video Generative Models"

This paper builds the largest open-source video generation model. It's trending on [Papers With Code](https://paperswithcode.com/) as of Dec 2024.

```python
annotate("https://arxiv.org/pdf/2412.03603v2")
```

![video](https://raw.githubusercontent.com/neuml/annotateai/master/images/video.png)

_Source: https://arxiv.org/pdf/2412.03603v2_

### Annotate paper "OpenDebateEvidence: A Massive-Scale Argument Mining and Summarization Dataset"

This paper was presented at the `38th Conference on Neural Information Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks`.

```python
annotate("https://arxiv.org/pdf/2406.14657")
```

![debate](https://raw.githubusercontent.com/neuml/annotateai/master/images/debate.png)

_Source: https://arxiv.org/pdf/2406.14657_

### Additional parameters

The default mode for an `annotate` instance is to automatically generate the key concepts to search for. But these concepts can be provided via the `keywords` parameter.

```python
annotate("https://arxiv.org/pdf/2005.11401", keywords=["hallucinations", "llm"])
```

This is useful for situations where we have a large batch of papers and we want it to identify a specific set of concepts to help with a review.

The progress bar can be disabled as follows:

```python
annotate("https://arxiv.org/pdf/2005.11401", progress=False)
```