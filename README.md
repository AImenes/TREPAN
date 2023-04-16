# TREPAN
Implementation of the TREPAN algorithm in Python.

https://proceedings.neurips.cc/paper_files/paper/1995/file/45f31d16b1058d586fc3be7207b58053-Paper.pdf

Read notebook for details.

Created in Python 3.9 environment.
Required libs: math, numpy, pandas, itertools, torch, sklearn, graphviz

To implement:
- Create predict method for trepan tree
- Fidelity method 
- Real stopping criteria with prob(p_c < (1 - epsilon)) < delta. Now we only have a cutoff value of the on p_c.

Known issues:
- m-of-n search only consideres 1-of-1 and 1-of-2 due to a bug. Currently working on this.
- Fidelity function doesnt compare tree to neural network, but neural network to neural network. Always returns max score of 1. Resulting in a FIFO expansion and not best first.

To run:
python run.py



