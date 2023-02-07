# This method is used to expand the tree, using a best first expansion.
# The best case is the node which has the greatest potential to increase the fidelity of the network.

# Fidelity = to what extent the decision tree mimics the Neural network, where 1 is perfect and 0 is everything wrong.
# instance = a single row of data

# Function used to evaluate a node n:
# f(n) = reach(n) x (1 - fidelity(n))
# the reach method is the estimated fraction of instances that reach the node n when passed through the tree
# the fidelity method is the estimated fidelity of the tree to the network for those instances.


#To be implemented
def reach(N):
    return None

def fidelity(instances):
    return None
#To be implemented

def best_first_expansion():
    return None
