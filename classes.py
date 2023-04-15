import math
import numpy as np
import pandas as pd
from itertools import combinations
import torch
import torch.nn as nn
from sklearn.neighbors import KernelDensity

class HeartDiseaseNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HeartDiseaseNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
            _, predicted = torch.max(output, 1)
        return predicted.numpy()

# Define the Oracle class
class Oracle:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.discrete_features, self.continuous_features = self._separate_features_by_type(self.X)
        self.discrete_distributions = self._model_discrete_features()
        self.continuous_kdes = self._model_continuous_features()

    def _separate_features_by_type(self, X, discrete_threshold=0.1):
        discrete_features = []
        continuous_features = []

        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            raise ValueError("X should be either a pandas DataFrame or a numpy array.")
            
        num_instances, num_features = X.shape

        for col in range(num_features):
            unique_values = np.unique(X[:, col])
            num_unique_values = len(unique_values)
            
            if (np.issubdtype(X[:, col].dtype, np.number) and num_unique_values / num_instances <= discrete_threshold) or not np.issubdtype(X[:, col].dtype, np.number):
                discrete_features.append(X[:, col])
            else:
                continuous_features.append(X[:, col])

        if len(discrete_features) > 0:
            discrete_features = np.column_stack(discrete_features)
        else:
            discrete_features = np.empty((num_instances, 0))

        if len(continuous_features) > 0:
            continuous_features = np.column_stack(continuous_features)
        else:
            continuous_features = np.empty((num_instances, 0))

        return discrete_features, continuous_features

    def _model_discrete_features(self):
        distributions = []
        for col in range(self.discrete_features.shape[1]):
            counts = np.bincount(self.discrete_features[:, col].astype(int))
            distributions.append(counts / counts.sum())
        return distributions

    def _model_continuous_features(self):
        kdes = []
        for col in range(self.continuous_features.shape[1]):
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(self.continuous_features[:, col].reshape(-1, 1))
            kdes.append(kde)
        return kdes

    def generate_instances(self, constraints, num_instances):
        new_instances = []

        for _ in range(num_instances):
            new_instance = []

            # Generate discrete features
            for col, distribution in enumerate(self.discrete_distributions):
                constrained_value = [constraint for constraint in constraints if constraint[0] == col]
                if constrained_value:
                    new_instance.append(constrained_value[0][1])
                else:
                    new_instance.append(np.random.choice(len(distribution), p=distribution))

            # Generate continuous features
            for col, kde in enumerate(self.continuous_kdes):
                constrained_value = [constraint for constraint in constraints if constraint[0] == col + self.discrete_features.shape[1]]
                if constrained_value:
                    new_instance.append(constrained_value[0][1])
                else:
                    new_instance.append(kde.sample()[0][0])

            new_instances.append(new_instance)

        return np.array(new_instances)

# Define the Node class
class Node:
    def __init__(self, training_examples, constraints, leaf, parent=None):
        self.leaf = leaf
        self.training_examples = training_examples
        self.constraints = constraints
        self.children = []
        self.parent = parent
        self.split = None
        self.label = None

    def _is_leaf(self):
        return len(self.children) == 0

# Define the MofN class
class MofN:
    def __init__(self, m, conditions, gain_ratio = 0):
        self.m = m
        self.conditions = conditions
        self.gain_ratio = gain_ratio
        self.outcomes = combinations(self.conditions)

# Define the TREPAN class
class TREPAN:
    def __init__(self, oracle, max_tree_size, max_conditions, max_children, cutoff, num_of_instances):
        self.oracle = oracle
        self.max_tree_size = max_tree_size
        self.max_conditions = max_conditions
        self.max_children = max_children
        self.current_amount_of_nodes = 0
        self.cutoff = cutoff
        self.number_of_instances_for_generation = num_of_instances
    
    def fit(self, X_true, y_true, features):

        # Define an empty queue
        queue = []

        # Use the oracle to predict y_predicted
        training_examples = self.oracle._predict(X_true)

        # Initialize root node as leaf
        self.root = Node(training_examples, {}, True)

        # Identify all possible candidate splits
        F = self._identify_candidate_splits(features)

        #Push node to queue
        queue.append(self.root)

        # Initialize best first expansion
        self._best_first_tree_expansion(queue)


    def _best_first_tree_expansion(self, queue, F):

        # 0. Run as long as there are still nodes in the queue and we havent reached limit
        while queue and self.current_amount_of_nodes < self.max_tree_size:
            
            # 1. Evaluate which node in the queue has the highest score, and we pop this one
            # The formula we use is f(n) = reach(n) * (1 - fidelity(n))
            N = self._get_best_scoring_node_in_queue(queue)

            # 2. Define F_N, which is the subset of all candidate splits which satisfies the current constraints
            F_N = self._extract_subset_of_candidate_splits(F, N.constraints)

            # 3. Query the oracle to get more instances
            X_from_oracle = self.oracle.generate_instances(N.constraints, num_instances = 20)

            # 4. Calculate gain ratio on all splits in F_N using gain ratio criterion
            # Let best_initial_split be the top scoring candidate split in F_N using N.training_examples and X_from_oracle
            best_initial_split, best_initial_gain_ratio = self._get_best_binary_split()

            # 5. Convert it a m-of-n format, that is a 1-of-{best_initial_split}
            best_binary_split = MofN(1, best_initial_split, best_initial_gain_ratio)

            # 6. If the gain ratio = 1, then we already have a splitting condition which cannot be improved
            # by a m-of-n search. Therefore, we only start the m-of-n search if we do not have a gain_ratio of 1.
            if not best_binary_split.gain_ratio == 1:
                
                best_split = self._calculate_best_m_of_n_split(best_binary_split, F_N)
                
            # If there is a max gain initial split
            else:
                best_split = best_binary_split

            # Set current node as an internal node
            N.leaf = False

            # 7. For every logical outcome of the m-of-n, we create a child node
            for new_constraints_c in best_split.outcomes:
                
                # 8. Append constraints from parent node N
                constraints_c = N.constraints + new_constraints_c
                training_examples_c = self._extract_training_examples_that_satisfy_constraints(N.training_examples)
                
                # 9. Generate new set of instances for evaluation. The number is the defined number in init for evaluation minus the number from training examples.
                instances_for_evaluation = self.model.generate_instances(constraints_c, self.number_of_instances_for_generation - (len(training_examples_c)))

                # Create a new child node as leaf node
                C = Node(training_examples_c, constraints_c, True, N)

                # Get the most common class prediction using the oracle
                most_common_class, p_c = self._calculate_proportion_of_most_common_class(instances_for_evaluation)
                
                # If proportion is larger than some cut-off value, let it be a leaf and assign target class
                if p_c >= self.cutoff:
                    C.label = most_common_class
                
                # Otherwise, append child node to the queue.
                else:
                    queue.append(C)
                
                



    def _calculate_best_m_of_n_split(self, best_binary_split, F_N):
     
        # Define terms
        threshold = best_binary_split.m # Corresponds to m
        current_number_of_conditions = len(best_binary_split.conditions) #Corresponds to len(n). Should be 1 when from binary split.
        best_gain = best_binary_split.gain_ratio
        best_conditions = best_binary_split.conditions

        # Include a new condition until we reach maximum allowance of conditions
        while current_number_of_conditions < self.max_conditions:
            
            # Increase threshold m. 
            for m in range(1, current_number_of_conditions + 1):

                # If the current structure m-of-current_number_of_conditions is less than max number of children.
                # Go to method defintion for explaination of this step.
                if self._get_child_count_for_MofN_structure(m, current_number_of_conditions) < self.max_children:

                    # If we are here, we are allowed to use this structure.
                    # For every candidate split that is not already selected
                    for candidate in F_N:
                        current_conditions = best_binary_split.conditions + candidate
                        current_gain = self._calculate_gain_ratio(current_conditions)
                        
                        # If the current configuration is better, save it as the best.
                        if current_gain > best_gain:
                            best_gain = current_gain
                            best_conditions = current_conditions
                            best_m = m

            current_number_of_conditions += 1

        return MofN(best_m, best_conditions, best_gain)
            
    def _get_child_count_for_MofN_structure(m, n):
        # From statistics, we are familiar with nCr, which is how many combinations of n items where r is selected, 
        # when order does not matter and repetitions are not allowed. We can use this here to calculate how 
        # many children a certain m-of-n structure will produce. Note that now, n is a number and not a condition for this calculation.
        # In our case, it will be nCm. For instance a 2-of-{A, B, C} will be a 2-of-3 structure with outcomes {A, B}, {A, C} and {B, C}
        # The number of children can be calculated: nCm = (n!) / ((n-m)! * m!)
        return ((math.factorial(n)) / (math.factorial(n-m) * math.factorial(m)))
    
    def _identify_candidate_splits(self, X):
        #...
        pass
        
    def _calculate_fidelity(self, y_true, y_pred):
        #...
        pass
    def _calculate_reach(self, X_true, constraints):
        #...
        pass

    def _calculate_node_score(self, node, X_true, y_true):
        #...
        pass

    def _calculate_proportion_of_most_common_class():
        pass

    def _extract_training_examples_that_satisfy_constraints():
        pass