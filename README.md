# TREPAN
Implementation of the TREPAN algorithm in Python

Created in Python 3.9 environment.
Required libs: math, numpy, pandas, itertools, torch, sklearn, graphviz

To implement:
- Create predict method for trepan tree
- Fidelity method 
- Real stopping criteria with prob(p_c < (1 - epsilon)) < delta. Now we only have a cutoff value of the on p_c.

Known issues:
- Fidelity function doesnt compare tree to neural network, but neural network to neural network. Always returns max score of 1. Resulting in a FIFO expansion and not best first.

To run:
python run.py



