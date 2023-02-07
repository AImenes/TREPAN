from classes.decisiontree import DecisionTree
from classes.node import Node
from methods.predict import predict


def trepan(X, y, features, oracle, **trepan_parameters):

    # Define the queue, Q
    Q = list()

    # Query the oracle for all training data in which the oracle was trained upon. Get labels.
    training_examples = predict(X, y, oracle)

    # Initialize the Tree, T, as a leaf node
    T = DecisionTree(**trepan_parameters)
    R = Node(True, True)      #Root node, R.
    T.append_node(R)

    # Enqueue root node, T, into Q.
    Q.append({'Node':R, 'examples':training_examples, 'constraints':[]})
    
    # While the queue is not empty and number of nodes are less than the limit
    while Q and (T.number_of_nodes < T.max_number_of_iternal_nodes):

        # Pop Node N from the head of the queue
        current_node = Q[0]
        Q.pop(0)

        # Define examples for node N

        # Define contraints for node N

        # Use features to build set of candidate splits

        # Use examplesN and call the Oracle(constraintsN)  to evaluate splits. Call the best binary split S.

        # Select best m-of-n split, S', using S as seed.

        # make N an internal node with split S'
        break

        #for each outcome, s, of S'

            # Define C, a new child of N

            # Constraints for C = ConstraintsN \union {S' = s}

            # Call Oracle(ConstraintsC) to determine if C should remain a leaf

            # if not

                # examplesC = members of examplesN with outcome s on split S'

                #append (C, examplesC, constraintsC) into queue, Q.


    return T

