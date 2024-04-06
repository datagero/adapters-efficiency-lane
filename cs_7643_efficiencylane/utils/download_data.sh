#!/bin/bash

# Define the output directory
output_dir="/Users/datagero/Documents/scratch/cs-7643-projectfiles/data"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Change directory to the output directory
cd "$output_dir" || exit


# Download files for sciie dataset
mkdir -p sciie
curl -Lo sciie/train.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/sciie/train.jsonl
curl -Lo sciie/dev.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/sciie/dev.jsonl
curl -Lo sciie/test.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/sciie/test.jsonl

# Download files for citation_intent dataset
mkdir -p citation_intent
curl -Lo citation_intent/train.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/citation_intent/train.jsonl
curl -Lo citation_intent/dev.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/citation_intent/dev.jsonl
curl -Lo citation_intent/test.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/citation_intent/test.jsonl
