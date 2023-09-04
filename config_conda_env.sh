#!/bin/bash -l

# Create the Conda environment
conda env create -f environment.yml --force -v

# Activate the environment
conda activate flowtx

# Install pip packages using the --no-cache-dir option
pip install --no-cache-dir -r requirements.txt
