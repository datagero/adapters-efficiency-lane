# Efficiency Lane: Task-Specific Adapters for RoBERTa on AdapterHub

### Initial Setup

#### Instructions are using mac.

This is optional. First clone the don’t stop pretraining library from github.
https://github.com/allenai/dont-stop-pretraining.git

Clone the following repository locally
https://github.gatech.edu/avizcaino3/CS-7643-EfficiencyLane.git

`conda env create -f environment.yml`

`conda activate dlgp`

### Download datasets

This will download all the datasets listed in the paper in the folder data/

`python download_datasets.py`

This might take a few minutes. You can alter the list in datasets.py if you want to download only a few datasets. The last dataset of amazon reviews takes the most time.

### Download models

We will also download the pretrained DAPT models from the Don’t Stop Pretraining model. We might adapt these too in addition to the base RoBERTa model.

allenai/cs_roberta_base
allenai/biomed_roberta_base
allenai/reviews_roberta_base
allenai/news_roberta_base

`python download_pretrained_models.py`

All the models will be downloaded to the folder pretrained_models/
This might take a few minutes.
