#!/bin/bash -l

# Create the Conda environment
conda env create -f environment.yml --force -v

# Activate the environment
conda activate flowtx

# Install pip packages using the --no-cache-dir option
pip install --no-cache-dir -r requirements.txt

# Install flowkit without deps because of sns version clash
pip install --no-deps git+https://github.com/hbhargava7/flowkit@hkb-add-autoread-fcs-from-flowjo

# Install FlowTx itself.
pip install -e .