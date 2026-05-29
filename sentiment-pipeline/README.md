# Sentiment analysis pipeline

## Apptainer image
Build apptainer image from `revised-container.def` with name `pytorcher2.sif`.

## Running on cluster

`sbatch run.slurm` -> don't forget to switch between `sent_analysis.py` and `sent_aggregation.py`, the steps are separate. First phase takes a few hours, the second one a few minutes.

## Using huggingface

The pipeline uses hardcoded output and input huggingface repos, for reproducibility, they should be changed by the user in the source code.
