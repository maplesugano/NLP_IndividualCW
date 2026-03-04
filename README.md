# NLP Individual Coursework
70016 (Natural Language Processing) 2026 Individual CW.

## Overview
This repository contains baseline experiments, exploratory analysis, local evaluation, and the best model checkpoint for the Don't Patronize Me task.

Best model weights are hosted on Hugging Face Hub:
https://huggingface.co/maplesugano/NLP_IndividualCW_best_model

## Repository Structure
- [dont_patronize_me.py](dont_patronize_me.py): Data loading and utilities.
- [BestModel/](BestModel/): Best model notebook and data splits.
- [dev.txt](dev.txt): Development split used by the best model.
- [test.txt](test.txt): Test split used by the best model.
- [data/](data/): Task data files.

## Setup
Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
The best model training and evaluation are contained in:
- [BestModel/baseline.ipynb](BestModel/baseline.ipynb)
- [BestModel/best_weightCE.ipynb](BestModel/best_weightCE.ipynb)

## Data
Raw and split data files are in [data/](data/). If you update file locations, check any hardcoded paths in [dont_patronize_me.py](dont_patronize_me.py) and the notebooks.