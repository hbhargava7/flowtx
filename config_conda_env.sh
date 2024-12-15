#!/bin/bash -l

# Create the Conda environment
conda env create -f environment.yml --force -v

# Activate the environment
conda activate flowtx

# Install pip packages using the --no-cache-dir option
# pip install --no-cache-dir -r requirements.txt

# Install dependencies, workaround to prevent installation from stopping if any one fails
cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip install --no-cache-dir

# Install flowkit without deps because of sns version clash
# pip install --no-deps git+https://github.com/hbhargava7/flowkit@hkb-add-autoread-fcs-from-flowjo

# Install FlowTx itself.
pip install -e .