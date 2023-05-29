# TREPAN
Implementation of the TREPAN algorithm in Python.

https://proceedings.neurips.cc/paper_files/paper/1995/file/45f31d16b1058d586fc3be7207b58053-Paper.pdf

## Installation

Created in Python 3.9 Conda environment.
Required libs: math, numpy, pandas, itertools, torch, sklearn, graphviz

## Code execution

To implement:
- Real stopping criteria with prob(p_c < (1 - epsilon)) < delta. Now we only have a cutoff value of the on p_c.

python ‘run_continuous’.py

The tree is also saved to file as 'trepan_tree.png' in the working directory.




