import math
import numpy as np
import pandas as pd
from itertools import combinations
import torch
import torch.nn as nn
from copy import deepcopy
from sklearn.neighbors import KernelDensity
from graphviz import Digraph

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class IrisNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IrisNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def fit(self, X, y, epochs=50, batch_size=16, lr=0.01): 

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Create data loaders
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):
            for data, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Print loss every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

    def predict(self, X):
        # Convert input to PyTorch tensor
        X = torch.tensor(X, dtype=torch.float32)

        # Get model predictions
        with torch.no_grad():
            output = self.forward(X)
            _, predicted_labels = torch.max(output, 1)

        return predicted_labels.numpy()

# Define the Oracle class
class Oracle:
    def __init__(self, model, X, y):
        self.model = model
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
        instances_created = 0

        while instances_created < num_instances:
            new_instance = []

            # Generate discrete features
            for col, distribution in enumerate(self.discrete_distributions):
                value = np.random.choice(len(distribution), p=distribution)
                new_instance.append(value)

            # Generate continuous features
            for col, kde in enumerate(self.continuous_kdes):
                
                value = round(kde.sample()[0][0], 1)
                new_instance.append(value)

            # Check if the generated instance satisfies all the constraints
            satisfies_constraints = True
            if constraints:
                for feature_idx, threshold, direction in constraints:
                    if direction == ">" and new_instance[feature_idx] <= threshold:
                        satisfies_constraints = False
                        break
                    if direction == "<=" and new_instance[feature_idx] > threshold:
                        satisfies_constraints = False
                        break

            # If the instance satisfies all constraints, add it to new_instances
            if satisfies_constraints:
                new_instances.append(new_instance)
                instances_created += 1

        return np.array(new_instances)

# Define the Node class
class Node:
    def __init__(self, training_examples, training_predictions, constraints, leaf, parent=None):
        self.leaf = leaf
        self.training_examples = training_examples
        self.training_predictions = training_predictions
        self.constraints = constraints
        self.children = []
        self.parent = parent
        self.split = None
        self.label = None
        self.score = 0

    def _is_leaf(self):
        return len(self.children) == 0

# Define the MofN class
class MofN:
    def __init__(self, m, conditions, gain_ratio = 0):
        self.m = m
        self.conditions = conditions
        self.gain_ratio = gain_ratio
        self.outcomes = None

# Define the TREPAN class
class TREPAN:
    def __init__(self, oracle, X, y, max_tree_size, max_conditions, max_children, cutoff, num_of_instances):
        self.oracle = oracle
        self.root = None
        self.features = X
        self.X_true = X
        self.y_true = y
        self.y_predicted = self.oracle.model.predict(self.X_true) 
        self.max_tree_size = max_tree_size
        self.max_conditions = max_conditions
        self.max_children = max_children
        self.current_amount_of_nodes = 0
        self.cutoff = cutoff
        self.number_of_instances_for_generation = num_of_instances
    
    def fit(self):

        # Define an empty queue
        queue = []

        # Use the oracle to predict y_predicted
        # This happens in the init process of TREPAN

        # Initialize root node as leaf
        self.root = Node(self.X_true, self.y_predicted, [], True)

        # Identify all possible candidate splits
        F = self._identify_candidate_splits(self.features)

        #Push node to queue
        queue.append(self.root)

        # Initialize best first expansion
        self._best_first_tree_expansion(queue, F)


    def _best_first_tree_expansion(self, queue, F):

        # 0. Run as long as there are still nodes in the queue and we havent reached limit
        while queue and self.current_amount_of_nodes < self.max_tree_size:
            
            # 1. Evaluate which node in the queue has the highest score, and we pop this one
            # The formula we use is f(n) = reach(n) * (1 - fidelity(n))
            N = self._get_best_scoring_node_in_queue(queue)
            queue.remove(N)

            # 2. Define F_N, which is the subset of all candidate splits which satisfies the current constraints
            F_N = self._extract_subset_of_candidate_splits(F, N.constraints)

            # 3. Query the oracle to get more instances
            X_from_oracle = self.oracle.generate_instances(N.constraints, num_instances = self.number_of_instances_for_generation - (len(N.training_examples)))
            y_from_oracle = self.oracle.model.predict(X_from_oracle)

            # 4. Calculate gain ratio on all splits in F_N using gain ratio criterion
            # Let best_initial_split be the top scoring candidate split in F_N using N.training_examples and X_from_oracle
            best_initial_split, best_initial_gain_ratio = self._get_best_binary_split(F_N, N, X_from_oracle, y_from_oracle)

            # 5. Convert it a m-of-n format, that is a 1-of-{best_initial_split}
            best_binary_split = MofN(1, best_initial_split, best_initial_gain_ratio)

            # 6. If the gain ratio = 1, then we already have a splitting condition which cannot be improved
            # by a m-of-n search. Therefore, we only start the m-of-n search if we do not have a gain_ratio of 1.
            if not best_binary_split.gain_ratio == 1:
                
                best_split = self._calculate_best_m_of_n_split(best_binary_split, F_N, N, X_from_oracle, y_from_oracle)
                
            # If there is a max gain initial split
            else:
                best_split = best_binary_split

            # Set current node as an internal node
            N.leaf = False
            N.split = best_split

            # Identify all logical outcomes
            best_split.outcomes = combinations(best_split.conditions, best_split.m)

            # 7. For every logical outcome of the m-of-n, we create a child node
            for new_constraints_c in best_split.outcomes:
                
                # 8. Append constraints from parent node N
                constraints_c = deepcopy(N.constraints) + deepcopy(list(new_constraints_c))

                # Update original training data using the new constraints
                child_mask = self._apply_constraints(constraints_c, N.training_examples)
                training_examples_c = deepcopy(N.training_examples[child_mask])
                training_predictions_c = deepcopy(N.training_predictions[child_mask])
                
                # 9. Generate new set of instances for evaluation. The number is the defined number in init for evaluation minus the number from training examples.
                instances_for_evaluation = self.oracle.generate_instances(constraints_c, self.number_of_instances_for_generation - (len(training_examples_c)))
                
                # Create a new child node as leaf node
                C = Node(training_examples_c, training_predictions_c, constraints_c, True, parent=N)
                N.children.append(C)
                self.current_amount_of_nodes += 1

                # Get the most common class prediction using the oracle
                most_common_class, p_c = self._most_common_class_proportion(instances_for_evaluation, training_examples_c, training_predictions_c)
                
                # If proportion is larger than some cut-off value, let it be a leaf and assign target class
                if p_c >= self.cutoff:
                    C.label = most_common_class
                
                # Otherwise, append child node to the queue.
                else:
                    queue.append(C)

    def _get_best_binary_split(self, F_N, node, X_from_oracle, y_from_oracle):
        best_gain_ratio = -1
        best_candidate_split = None
        X = np.vstack((node.training_examples, X_from_oracle))
        y = np.concatenate((node.training_predictions, y_from_oracle))

        for candidate in F_N:
            gain_ratio = self._calculate_gain_ratio(X, y, [candidate])

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_candidate_split = [candidate]

        return best_candidate_split, best_gain_ratio
                
    def _calculate_best_m_of_n_split(self, best_binary_split, F_N, node, X_from_oracle, y_from_oracle):
        X = np.vstack((node.training_examples, X_from_oracle))
        y = np.concatenate((node.training_predictions, y_from_oracle))

        # Define terms
        best_m = best_binary_split.m # Corresponds to m
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
                        current_conditions = best_binary_split.conditions + [candidate]
                        current_gain = self._calculate_gain_ratio(X, y, current_conditions)
                        
                        # If the current configuration is better, save it as the best.
                        if current_gain > best_gain:
                            best_gain = current_gain
                            best_conditions = current_conditions
                            best_m = m

            current_number_of_conditions += 1

        return MofN(best_m, best_conditions, best_gain)
            
    def _get_child_count_for_MofN_structure(self, m, n):
        # From statistics, we are familiar with nCr, which is how many combinations of n items where r is selected, 
        # when order does not matter and repetitions are not allowed. We can use this here to calculate how 
        # many children a certain m-of-n structure will produce. Note that now, n is a number and not a condition for this calculation.
        # In our case, it will be nCm. For instance a 2-of-{A, B, C} will be a 2-of-3 structure with outcomes {A, B}, {A, C} and {B, C}
        # The number of children can be calculated: nCm = (n!) / ((n-m)! * m!)
        return ((math.factorial(n)) / (math.factorial(n-m) * math.factorial(m)))
    
    def _identify_candidate_splits(self, X):
        """
        Identify all possible candidate splits for all features in the input feature matrix X.

        Parameters
        ----------
        X : numpy.ndarray
            The input feature matrix.

        Returns
        -------
        candidate_splits : list of tuples
            A list of candidate splits, where each split is represented as a tuple (feature_index, threshold).
        """
        # Get the number of instances and features
        num_instances, num_features = X.shape
        
        # Initialize an empty list to store candidate splits
        candidate_splits = []

        # Iterate over each feature column in the input matrix X
        for col in range(num_features):
            # Find the unique values for the current feature
            unique_values = np.unique(X[:, col])
            
            # If there's only one unique value, continue to the next feature
            if len(unique_values) == 1:
                continue

            # If there are more than two unique values, calculate thresholds as the midpoint between adjacent unique values
            if len(unique_values) > 2:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
            else:
                thresholds = unique_values

            # Append the feature index and candidate split threshold to the list of candidate splits
            for threshold in thresholds:
                candidate_splits.append((col, round(threshold, 2), '<='))
                candidate_splits.append((col, round(threshold, 2), '>'))

        return candidate_splits
        
    def _get_best_scoring_node_in_queue(self, queue):
        best_score = 0
        best_node = None

        for node in queue:
            score = self._calculate_node_score(node)
            node.score = score

            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node

    def _calculate_node_score(self, node):
        """
        Calculate the score for a given node.

        Parameters
        ----------
        node : TrepanNode
            The node for which to calculate the score.
        X_true : numpy.ndarray
            The input feature matrix.
        y_true : numpy.ndarray
            The true labels.

        Returns
        -------
        score : float
            The node score.
        """
        # Apply the constraints of the node
        mask = self._apply_constraints(node.constraints, node.training_examples)

        # Apply mask
        X_filtered = node.training_examples[mask]
        y_filtered = node.training_predictions[mask]

        # Calculate the fidelity
        y_pred = self.oracle.model.predict(X_filtered)
        fidelity = self._calculate_fidelity(y_filtered, y_pred)

        # Calculate the reach
        reach = self._calculate_reach(node.constraints)

        # Calculate the node score
        score = reach * fidelity
        return score

    def _calculate_fidelity(self, y_true, y_pred):
        """
        Calculate the fidelity of the predicted values compared to the true values.

        Parameters
        ----------
        y_true : numpy.ndarray
            The true labels.
        y_pred : numpy.ndarray
            The predicted labels.

        Returns
        -------
        fidelity : float
            The fidelity score.
        """
        assert y_true.shape == y_pred.shape, "y_true and y_pred should have the same shape."
        return np.mean(y_true == y_pred)

    def _calculate_reach(self, constraints):
        """
        Calculate the reach of a given set of instances given a set of constraints.

        Parameters
        ----------
        X_true : numpy.ndarray
            The input feature matrix.
        constraints : list of tuples
            The constraints to apply.

        Returns
        -------
        reach : float
            The reach score.
        """
        num_instances, _ = self.X_true.shape
        num_instances_satisfying_constraints = 0

        if not constraints:
            return 1

        for instance in self.X_true:
            satisfies_constraints = True
            for feature_index, threshold, direction in constraints:
                if (direction == 'less' and not (instance[feature_index] <= threshold)) or (direction == 'greater' and not (instance[feature_index] > threshold)):
                    satisfies_constraints = False
                    break
            if satisfies_constraints:
                num_instances_satisfying_constraints += 1

        reach = num_instances_satisfying_constraints / num_instances
        return reach

    def _apply_constraints(self, constraints, X):
        # Create a boolean mask with the same length as the number of instances in X, initialized with True values
        mask = np.ones(len(X), dtype=bool)

        # Iterate over each constraint in the list of constraints
        for col, threshold, direction in constraints:
            if direction == '<=':
                # Create a boolean mask for instances where the feature value at column 'col' is less than or equal to the threshold
                current_mask = X[:, col] <= threshold
            
            elif direction == '>':
                current_mask = X[:, col] > threshold
            
            else:
                raise SyntaxError("Wrong syntax for direction in tuple. Should be '<=' or '>'.")
            
            # Apply the current mask to the main mask using the AND operation
            mask = mask & current_mask

        # Return the mask
        return mask
    
    def _most_common_class_proportion(self, X_from_oracle, training_examples_c, training_predictions_c):
            # Predict the targets on the instances
        y_pred = self.oracle.model.predict(X_from_oracle)

        y = np.concatenate((training_predictions_c, y_pred))

        # Count the occurrences of each class
        unique_classes, counts = np.unique(y, return_counts=True)

        # Find the index of the most common class
        most_common_class_idx = np.argmax(counts)

        # Calculate the proportion of the most common class
        most_common_class = unique_classes[most_common_class_idx]
        proportion = counts[most_common_class_idx] / y.size

        return most_common_class, proportion

    def _extract_subset_of_candidate_splits(self, F, constraints):
        """
        Extracts the subset of candidate splits that satisfy the list of constraints.

        :param F: list of candidate splits (tuples of structure (feature_idx, threshold))
        :param constraints: list of constraints (tuples of type (feature_idx, threshold, direction))
        :return: list of candidate splits that satisfy the constraints
        """

        if not constraints:
            return F

        def satisfies_constraints(candidate_split):
            feature_idx, threshold, direction = candidate_split

            for constraint_feature_idx, constraint_threshold, direction in constraints:
                if feature_idx == constraint_feature_idx:
                    if direction == "<=" and threshold > constraint_threshold:
                        return False
                    elif direction == ">" and threshold <= constraint_threshold:
                        return False

            return True

        filtered_splits = [split for split in F if satisfies_constraints(split)]

        return filtered_splits

    def _calculate_gain_ratio(self, X, y, constraints):
        
        if not constraints:
            mask = np.ones(len(X), dtype=bool)
        else: 
            mask = self._apply_constraints(constraints, X)
        
        
        y_entropy = self._calculate_entropy(y)

        y_left = y[mask]
        y_right = y[~mask]

        # Calculate the entropy for both splits
        left_entropy = self._calculate_entropy(y_left)
        right_entropy = self._calculate_entropy(y_right)

        # Calculate the weighted average of the entropies
        left_weight = len(y_left) / len(y)
        right_weight = len(y_right) / len(y)
        avg_entropy = left_weight * left_entropy + right_weight * right_entropy

        # Calculate information gain
        info_gain = y_entropy - avg_entropy

        # Calculate intrinsic value
        intrinsic_value = -left_weight * np.log2(left_weight) - right_weight * np.log2(right_weight) if left_weight > 0 and right_weight > 0 else 0

        # Calculate gain ratio
        gain_ratio = info_gain / intrinsic_value if intrinsic_value != 0 else 0

        return gain_ratio

    def _calculate_entropy(self, y):
            num_instances = len(y)
            unique_labels, label_counts = np.unique(y, return_counts=True)
            probabilities = label_counts / num_instances
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return entropy
    

    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root

        # Print the current node
        if node.leaf:
            print("  " * level + f"Leaf (Class: {node.label}, Constraints: {[condition for condition in node.constraints]})")
        else:
            print("  " * level + f"Node (Split: {node.split.m}-of-{[condition for condition in node.split.conditions]}, Gain Ratio: {node.split.gain_ratio:.2f})")

        # Recursively print the children
        for child in node.children:
            self.print_tree(child, level + 1)

    def to_graphviz(self, node=None, graph=None, parent=None):
        if node is None:
            node = self.root

        if graph is None:
            graph = Digraph("TREPAN_Tree", format="png")
            graph.attr(rankdir="TB")

        node_id = f"{id(node)}"
        if node.leaf:
            label = f"Leaf\n(Class: {node.label},\nConstraints: {[condition for condition in node.constraints]})"
        else:
            label = f"Node\n(Split: {node.split.m}-of-{[condition for condition in node.split.conditions]},\nGain Ratio: {node.split.gain_ratio:.2f})"

        graph.node(node_id, label=label)

        if parent is not None:
            graph.edge(f"{id(parent)}", node_id)

        for child in node.children:
            self.to_graphviz(child, graph, node)

        return graph