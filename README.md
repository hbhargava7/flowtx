# flowtx

Toolkit for flow cytometry data analysis

Internal reference: https://www.notion.so/hershbhargava/PMOC-Flowtx-154f70a8edc78081a563e27f4078f0d3?pvs=4

### Installation and Usage

After the cloning the repository you can build the FlowTx conda environment using the script

```bash

./config_conda_env.sh

```

This will create a conda environment named `flowtx` with all the necessary packages to run the scripts in this repository.

### Notes

Note on dependencies: Flowtx relies on `whitews/flowkit` to interface with FlowJo files. When FlowTx was initially built