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
from sklearn.model_selection import train_test_split

class IrisNN(nn.Module):
    """
    A neural network model for the Iris dataset classification task.

    This class extends the PyTorch `nn.Module` and implements a simple feedforward neural network
    with a single hidden layer for the Iris dataset classification. It includes methods for training,
    fitting, and predicting using the neural network.

    Attributes:
        fc1 (nn.Linear): First fully connected layer, mapping input features to hidden dimensions.
        relu (nn.ReLU): Rectified Linear Unit activation function.
        fc2 (nn.Linear): Second fully connected layer, mapping hidden dimensions to output dimensions.
        softmax (nn.Softmax): Softmax activation function for converting logits to probabilities.

    Example:
        iris_nn = IrisNN(input_dim=4, hidden_dim=10, output_dim=3)
        iris_nn.fit(X, y, epochs=100, batch_size=16, lr=0.01)
        y_pred = iris_nn.predict(X)
    """
    def __init__(self, input_dim, output_dim):
        super(IrisNN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=12)
        self.relu = nn.ReLU()
        self.output = nn.Linear(in_features=12, out_features=output_dim)
 
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x

    def fit(self, X, y, epochs=30, batch_size=20, lr=0.01): 

        # Convert to PyTorch tensors
        X_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.long)

        # Create data loaders
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

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
    """
    This is the oracle class used to bind the tree expansion and neural network together. We use this to generate more instances of values, model feature distribution and to keep track of the neural network.
    """
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.discrete_features, self.continuous_features = self._separate_features_by_type(self.X)
        self.discrete_distributions = self._model_discrete_features()
        self.continuous_kdes = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(self.continuous_features)

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

    def generate_instances(self, hard_constraints, current_split, num_instances):
        new_instances = []
        instances_created = 0

        while instances_created < num_instances:
            new_instance = []

            # Generate discrete features
            for col, distribution in enumerate(self.discrete_distributions):
                value = np.random.choice(len(distribution), p=distribution)
                new_instance.append(value)

            # Generate continuous features
            #for col, kde in enumerate(self.continuous_kdes):
            new_sample = self.continuous_kdes.sample()[0]

            # Round the values and append them to new_instance
            new_instance = [round(value, 1) for value in new_sample]

            # Check if the generated instance satisfies all the constraints
            satisfies_constraints = True

            if hard_constraints:
                for constraint in hard_constraints:
                    m = constraint.m
                    conditions = constraint.conditions
                    satisfied_conditions_count = 0

                    for feature_idx, threshold, direction in conditions:
                        if direction == "<=" and new_instance[feature_idx] <= threshold:
                            satisfied_conditions_count += 1
                        if direction == ">" and new_instance[feature_idx] > threshold:
                            satisfied_conditions_count += 1

                    if constraint.satisfied and satisfied_conditions_count < m:
                        satisfies_constraints = False
                        break
                    elif not constraint.satisfied and satisfied_conditions_count >= m:
                        satisfies_constraints = False
                        break

            if satisfies_constraints and current_split:
                # Check if the generated instance satisfies the current split
                m = current_split.m
                conditions = current_split.conditions
                satisfied_conditions_count = 0

                for feature_idx, threshold, direction in conditions:
                    if direction == "<=" and new_instance[feature_idx] <= threshold:
                        satisfied_conditions_count += 1
                    if direction == ">" and new_instance[feature_idx] > threshold:
                        satisfied_conditions_count += 1

                if current_split.satisfied and satisfied_conditions_count >= m:
                    new_instances.append(new_instance)
                    instances_created += 1
                elif not current_split.satisfied and satisfied_conditions_count < m:
                    new_instances.append(new_instance)
                    instances_created += 1
            elif satisfies_constraints and not current_split:
                new_instances.append(new_instance)
                instances_created += 1

        return np.array(new_instances)

# Define the Node class
class Node:
    def __init__(self, training_examples, training_predictions, constraints, leaf, parent=None, reach=1):
        self.leaf = leaf
        self.training_examples = training_examples
        self.training_predictions = training_predictions
        self.hard_constraints = constraints
        self.split = None
        self.children = []
        self.parent = parent
        self.label = np.argmax(np.bincount(self.training_predictions))
        self.score = 0
        self.reach = reach
        self.fidelity = 0
        self.available_features = self._find_available_features()

    def _is_leaf(self):
        return len(self.children) == 0
    
    def _find_available_features(self):
        # Initialize feature count dictionary
        feature_count = {i: 0 for i in range(self.training_examples.shape[1])}

        # Count constraints for each feature
        for constraint in self.hard_constraints:
            # Tuple of type (feature_idz, threshold_value, ge or lt)
            for condition in constraint.conditions:
                feature_count[condition[0]] += 1

        # Determine available features (those with less than 2 constraints)
        return [feature for feature, count in feature_count.items() if count < 2]



# Define the MofN class
class MofN:
    def __init__(self, m, conditions, gain_ratio = 0):
        self.m = m
        self.conditions = conditions
        self.gain_ratio = gain_ratio
        self.satisfied = True

# Define the TREPAN class
class TREPAN:
    def __init__(self, oracle, X, y, max_tree_size, max_conditions, max_children, cutoff, num_of_instances):
        self.oracle = oracle
        self.root = None
        self.features = X
        self.X_true = X
        self.y_true = y
        self.y_predicted = self.oracle.model.predict(self.X_true) 
        self.length = len(self.y_true)
        self.max_tree_size = max_tree_size
        self.max_conditions = max_conditions
        self.max_children = max_children
        self.current_amount_of_nodes = 0
        self.cutoff = cutoff
        self.S_min = len(self.X_true) // 10
        self.number_of_instances_for_generation = num_of_instances
        self.epsilon = 1e-9
    
    def predict(self, X):
        """
        Predict the class labels for the given instances using the TREPAN decision tree.

        Args:
            X (numpy.ndarray): An array of instances for which class labels need to be predicted.

        Returns:
            numpy.ndarray: An array of predicted class labels for the input instances.
        """

        def traverse_tree(node, instance):
            """
            Traverse the tree recursively to find the leaf node corresponding to the given instance.

            Args:
                node (Node): The current node being traversed.
                instance (numpy.ndarray): A single instance for which a leaf node needs to be found.

            Returns:
                Node: The leaf node corresponding to the given instance.
            """

            # If the current node is a leaf, return the node
            if node.leaf:
                return node

            # If the current node is an internal node, traverse its children based on the m-of-n conditions
            else:
                for child in node.children:
                    if self._satisfies_m_of_n_conditions(child.hard_constraints, instance):
                        return traverse_tree(child, instance)


        # Initialize an array to store predicted class labels
        predictions = np.empty(X.shape[0], dtype=int)

        # Iterate through the instances and predict class labels
        for idx, instance in enumerate(X):
            leaf_node = traverse_tree(self.root, instance)
            predictions[idx] = leaf_node.label

        return predictions

    def fit(self):

        # Define an empty queue
        queue = []

        # Use the oracle to predict y_predicted
        # This happens in the init process of TREPAN

        # Initialize root node as leaf
        self.root = Node(self.X_true, self.y_predicted, [], True, None, 1)
        self._calculate_node_score(self.root)

        # Identify all possible candidate splits
        F = self._identify_candidate_splits(self.features)

        #Push node to queue
        queue.append(self.root)

        # Initialize best first expansion
        self._best_first_tree_expansion(queue, F)

    def _best_first_tree_expansion(self, queue, F):
        """
        Expands the decision tree by selecting the best node and its corresponding split.
        This function iteratively evaluates nodes in the queue, computes their gain ratio, and
        generates child nodes until the maximum tree size is reached or the queue is empty.

        Args:
            queue (list): List of nodes to be evaluated and expanded.
            F (list): List of candidate splits to be considered for each node.
        """

        # 0. Run as long as there are still nodes in the queue and we havent reached limit
        while queue and self.current_amount_of_nodes < self.max_tree_size:
            
            # 1. Evaluate which node in the queue has the highest score, and we pop this one
            # The formula we use is f(n) = reach(n) * (1 - fidelity(n))
            # Todo: Fix fidelity to compare towards tree.
            N = self._get_best_scoring_node_in_queue(queue)
            queue.remove(N)

            # 2. Define F_N, which is the subset of all candidate splits which satisfies the current constraints
            F_N = self._extract_subset_of_candidate_splits(F, N.hard_constraints)

            # 3. If the split has less than half of the original length, generate the remainings in order to get to 100.
            if len(N.training_examples) < self.number_of_instances_for_generation:
                X_from_oracle = self.oracle.generate_instances(N.hard_constraints, None, num_instances = (self.number_of_instances_for_generation - len(N.training_examples)))
                y_from_oracle = self.oracle.model.predict(X_from_oracle)
            else:
                X_from_oracle = np.empty((0, N.training_examples.shape[1]))
                y_from_oracle = np.empty((0,), dtype=np.int64)

            # 4. Calculate gain ratio on all splits in F_N using gain ratio criterion
            # Let best_initial_split be the top scoring candidate split in F_N using N.training_examples and X_from_oracle
            best_initial_split, best_initial_gain_ratio = self._get_best_binary_split(F_N, N, X_from_oracle, y_from_oracle)

            # 5. Convert it a m-of-n format, that is a 1-of-{best_initial_split}
            best_binary_split = MofN(1, best_initial_split, best_initial_gain_ratio)

            # 6. If the gain ratio = 1, then we already have a splitting condition which cannot be improved
            # by a m-of-n search. Therefore, we only start the m-of-n search if we do not have a gain_ratio of 1.
            if not best_binary_split.gain_ratio >= 1:
                
                best_split = self._calculate_best_m_of_n_split(best_binary_split, F_N, N, X_from_oracle, y_from_oracle)
                
            # If there is a max gain initial split
            else:
                best_split = best_binary_split

            # 8. Set current node as an internal node
            N.leaf = False
            N.split = best_split


            # For every logical outcome of the m-of-n, we create a child node
            #for new_constraints_c in best_split.outcomes:
            for split_satisfied in [True, False]:
                
                # 11. Append constraints from parent node N
                best_split.satisfied = split_satisfied
                constraints_c = deepcopy(N.hard_constraints) + deepcopy([best_split])

                # 12. Update original training data using the new constraints
                child_mask = self._apply_m_of_n_constraints(constraints_c, N.training_examples)

                # if there indeed is a split, and not one side is empty
                if not (True in child_mask and False in child_mask):
                    N.leaf = True
                    break    

                # Get new sets of entities
                training_examples_c = deepcopy(N.training_examples[child_mask])
                training_predictions_c = deepcopy(N.training_predictions[child_mask])

                
                # 13. Generate new set of instances for evaluation. The number is the defined number in init for evaluation minus the number from training examples.
                #if len(training_examples_c) < self.length:
                instances_for_evaluation = self.oracle.generate_instances(N.hard_constraints, best_split, self.length)
                #else:
                #    instances_for_evaluation = np.array([], dtype=np.int64)

                # 14. Create a new child node as leaf node, define it as child of N, and add to node count (for stopping criteria).
                if training_examples_c.size != 0:
                    C = Node(training_examples_c, training_predictions_c, constraints_c, True, parent=N)
    
                    N.children.append(C)
                    self.current_amount_of_nodes += 1

                    # 15. Get the most common class prediction using the oracle
                    most_common_class, p_c = self._most_common_class_proportion(instances_for_evaluation, training_predictions_c)
                    
                    # 16. If proportion is larger than some cut-off value, let it be a leaf and assign target class
                    C.label = most_common_class

                    # Calculate the score of the node
                    self._calculate_node_score(C)

                    # Calculate
                    
                    # Otherwise, append child node to the queue.
                    if p_c < self.cutoff:
                        queue.append(C)
                
                        

        if queue:
            for node in queue:
                most_common_class, p_c = self._most_common_class_proportion(np.array([]), node.training_examples, node.training_predictions)
                node.label = most_common_class

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

    def _get_best_binary_split(self, F_N, node, X_from_oracle, y_from_oracle):
        best_gain_ratio = -1
        best_candidate_split = None
        X = np.vstack((node.training_examples, X_from_oracle))
        y = np.concatenate((node.training_predictions, y_from_oracle))

        for candidate in F_N:
            if candidate[0] in node.available_features:

                gain_ratio = self._calculate_gain_ratio(X, y, [candidate])

                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_candidate_split = [candidate]

        if best_gain_ratio > 1:
            best_gain_ratio = 1

        return best_candidate_split, best_gain_ratio
                
    def _calculate_best_m_of_n_split(self, best_binary_split, F_N, node, X_from_oracle, y_from_oracle):
        X = np.vstack((node.training_examples, X_from_oracle))
        y = np.concatenate((node.training_predictions, y_from_oracle))

        best_m_of_n_split = best_binary_split
        current_conditions = list(best_binary_split.conditions)

        def condition_exists(conditions, candidate):
            for cond in conditions:
                if cond[0] == candidate[0] and cond[2] == candidate[2]:
                    return True
            return False
        
        def condition_illegal(conditions, candidate):
            for cond in conditions:
                if cond[0] == candidate[0] and cond[2] == candidate[2]:
                    return True
            return False
    
        while len(current_conditions) < self.max_conditions:
            # Find the best condition to add from F_N
            best_new_condition = None
            best_gain_ratio = 0
            best_m = deepcopy(best_m_of_n_split.m)

            for candidate in F_N:
                if (candidate not in current_conditions) and (not condition_exists(current_conditions, candidate)) and (candidate[0] in node.available_features):
                    # Try adding the candidate to the conditions
                    extended_conditions = current_conditions + [candidate]

                    # Calculate gain ratio for m-of-n+1
                    gain_ratio_m_of_n_plus_1 = self._calculate_gain_ratio_m_of_n(X, y, extended_conditions, best_m)
                    # Calculate gain ratio for m+1-of-n+1
                    gain_ratio_m_plus_1_of_n_plus_1 = self._calculate_gain_ratio_m_of_n(X, y, extended_conditions, best_m + 1)

                    if gain_ratio_m_plus_1_of_n_plus_1 > gain_ratio_m_of_n_plus_1:
                        candidate_gain_ratio = gain_ratio_m_plus_1_of_n_plus_1
                        candidate_m = best_m + 1
                    else:
                        candidate_gain_ratio = gain_ratio_m_of_n_plus_1
                        candidate_m = best_m

                    #decimal error fix. Could be 1.00000002 for instance.
                    if candidate_gain_ratio > 1:
                        candidate_gain_ratio = 1

                    # To make sure that the added complexity added by increasing the search has some substantial increase in gain, we add a minimum increase of 0.01.
                    if (candidate_gain_ratio - 0.01) > best_gain_ratio:
                        best_new_condition = candidate
                        best_gain_ratio = candidate_gain_ratio
                        best_m = candidate_m

            # Add the best new condition to the current conditions
            if best_new_condition is not None:
                current_conditions.append(best_new_condition)

                # Update the best_m_of_n_split with the new conditions and gain ratio
                if best_gain_ratio > best_m_of_n_split.gain_ratio:
                    best_m_of_n_split = MofN(best_m, current_conditions, best_gain_ratio)

            # Break the loop if the gain ratio is 1.0 or if the number of children exceeds the maximum allowed
            if best_gain_ratio >= 1.0 or self._get_child_count_for_MofN_structure(best_m, len(current_conditions)) >= self.max_children:
                break

        return best_m_of_n_split

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
    
    def _apply_m_of_n_constraints(self, mofn_split_list, X):
        overall_mask = np.ones(len(X), dtype=bool)

        # Iterate over each MofN split object in the list
        for mofn_split in mofn_split_list:
            m = mofn_split.m
            conditions = mofn_split.conditions
            satisfied = mofn_split.satisfied

            # Initialize an array to count the number of satisfied conditions for each instance
            satisfied_conditions_count = np.zeros(len(X), dtype=int)

            for feature_idx, threshold_value, less_than_or_greater in conditions:
                if less_than_or_greater == '<=':
                    satisfied_conditions_count += (X[:, feature_idx] <= threshold_value)
                else:
                    satisfied_conditions_count += (X[:, feature_idx] > threshold_value)

            # Get the mask for instances satisfying the m-of-n split
            m_of_n_mask = satisfied_conditions_count >= m
            
            if satisfied:
                overall_mask &= m_of_n_mask
            else:
                overall_mask &= (~m_of_n_mask)

        return overall_mask


    def _get_child_count_for_MofN_structure(self, m, n):
        # From statistics, we are familiar with nCr, which is how many combinations of n items where r is selected, 
        # when order does not matter and repetitions are not allowed. We can use this here to calculate how 
        # many children a certain m-of-n structure will produce. Note that now, n is a number and not a condition for this calculation.
        # In our case, it will be nCm. For instance a 2-of-{A, B, C} will be a 2-of-3 structure with outcomes {A, B}, {A, C} and {B, C}
        # The number of children can be calculated: nCm = (n!) / ((n-m)! * m!)
        return ((math.factorial(n)) / (math.factorial(n-m) * math.factorial(m)))
    
    def _get_best_scoring_node_in_queue(self, queue):
        best_score = float('-inf')
        best_node = None

        for node in queue:
            if node.score > best_score:
                best_score = node.score
                best_node = node
        
        return best_node

    def _calculate_node_score(self, node):
        """
        Calculate the score for a given node.
        
        !! This doesnt work yet !!

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
        # Calculate the reach
        self._calculate_reach(node)

        # Calculate the fidelity
        self._calculate_fidelity(node)

        # Calculate the node score
        node.score = node.reach * (1 - node.fidelity)


    def _calculate_fidelity(self, node):
        temp_tree_prediction = self.predict(node.training_examples)
        agreed_predictions = np.sum(node.training_predictions == temp_tree_prediction)
        node.fidelity = agreed_predictions / len(node.training_predictions)

    def _calculate_reach(self, node):
        if node.parent:
            node.reach = node.parent.reach * (len(node.training_examples) / self.length)
        else:
            node.reach = (len(node.training_examples) / self.length)

    
    def _most_common_class_proportion(self, X_from_oracle, training_predictions_c):
        # Predict the targets on the instances
        if X_from_oracle.size > 0:
            y = self.oracle.model.predict(X_from_oracle)
            #y = np.concatenate((training_predictions_c, y_pred))
        else:
            y = training_predictions_c    
                #y = training_predictions_c
        #while len(y) < self.S_min:

        if y.size == 0:
            return None
        
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
        :param constraints: list of MofN constraints
        :return: list of candidate splits that satisfy the constraints
        """

        if not constraints:
            return F

        def satisfies_constraints(candidate_split):
            feature_idx, threshold, direction = candidate_split

            for constraint in constraints:
                constraint_m = constraint.m
                
                # First we check if the number of conditions equals the threshold, like 1-of-1, 2-of-2 etc. These are the only scenarios where we can safely eliminate from the search space.
                if len(constraint.conditions) == constraint.m:

                    for constraint_feature_idx, constraint_threshold, constraint_direction in constraint.conditions:
                        if feature_idx == constraint_feature_idx:
                            if constraint_direction == "<=" and threshold > constraint_threshold:
                                return False
                            elif constraint_direction == ">" and threshold <= constraint_threshold:
                                return False

            return True

        result = [candidate_split for candidate_split in F if satisfies_constraints(candidate_split)]
        return result

    def _satisfies_m_of_n_conditions(self, mofn_constraints, instance):
        """
        Check if a given instance satisfies the m-of-n conditions of a specific node.

        Args:
        mofn_constraints (list of MofN objects): List of MofN constraints.
        instance (numpy array): The instance we want to check against the constraints.

        Returns:
        bool: True if the instance satisfies the m-of-n conditions, False otherwise.
        """

        all_constraints_satisfied = True

        for mofn_constraint in mofn_constraints:
            m = mofn_constraint.m
            conditions = mofn_constraint.conditions
            satisfied_conditions_count = 0

            for feature_idx, threshold, direction in conditions:
                if direction == "<=":
                    if instance[feature_idx] <= threshold:
                        satisfied_conditions_count += 1
                elif direction == ">":
                    if instance[feature_idx] > threshold:
                        satisfied_conditions_count += 1

            if satisfied_conditions_count >= m:
                constraint_satisfied = mofn_constraint.satisfied
            else:
                constraint_satisfied = not mofn_constraint.satisfied

            all_constraints_satisfied = all_constraints_satisfied and constraint_satisfied

        return all_constraints_satisfied

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
        intrinsic_value = -left_weight * np.log2(left_weight + self.epsilon) - right_weight * np.log2(right_weight + self.epsilon) if left_weight > 0 and right_weight > 0 else 0

        # Calculate gain ratio
        gain_ratio = info_gain / intrinsic_value if intrinsic_value != 0 else 0

        return gain_ratio

    def _calculate_gain_ratio_m_of_n(self, X, y, new_proposed_conditions, m):
        # Calculate the entropy of the parent node before splitting
        parent_entropy = self._calculate_entropy(y)

        gain_ratios = []

        # Iterate over each logical outcome of the m-of-n split using itertools.combinations
        for outcome in combinations(new_proposed_conditions, m):
            new_constraints = list(outcome)
            weighted_entropy = 0
            intrinsic_value = 0

            # Apply the constraints to the dataset
            child_mask = self._apply_constraints(new_constraints, X)
            child_examples = X[child_mask]
            child_predictions = y[child_mask]
            not_applicable = X[~child_mask]
            not_applicable_predictions = y[~child_mask]

            # Calculate the entropy of the resulting child node
            child_entropy = self._calculate_entropy(child_predictions)
            not_child_entropy = self._calculate_entropy(not_applicable_predictions)

            # Calculate the proportion of instances that fall into the child node
            child_proportion = len(child_examples) / len(X)

            # Update the weighted child entropy sum and intrinsic value sum
            weighted_entropy += ((child_proportion * child_entropy) + ((1 - child_proportion) * not_child_entropy))
            intrinsic_value -= ((child_proportion * np.log2(child_proportion + self.epsilon)) + ((1 - child_proportion) * np.log2(1 - child_proportion + self.epsilon)))
        
            # Calculate the information gain and gain ratio
            information_gain = parent_entropy - weighted_entropy
            gain_ratio = information_gain / intrinsic_value
        
            gain_ratios.append(gain_ratio)

        return np.mean(gain_ratios)

    def _calculate_intrinsic_value(self, y):
        num_instances = len(y)
        unique_labels, label_counts = np.unique(y, return_counts=True)
        proportions = label_counts / num_instances
        
        # Calculate the intrinsic value (split information)
        intrinsic_value = -np.sum(proportions * np.log2(proportions + self.epsilon))
        
        return intrinsic_value

    def _calculate_entropy(self, y):
            num_instances = len(y)
            unique_labels, label_counts = np.unique(y, return_counts=True)
            probabilities = label_counts / num_instances
            entropy = -np.sum(probabilities * np.log2(probabilities + self.epsilon))
            return entropy
    
    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root

        # Print the current node
        if node.leaf:
            print("  " * level + f"Leaf (Class: {node.label}, Constraints: ", end="")
            for mofn in node.hard_constraints:
                satisfaction = "Satisfied" if mofn.satisfied else "Not Satisfied"
                print(f"{satisfaction} {mofn.m}-of-{[condition for condition in mofn.conditions]}", end=", ")
            print(")")

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
            constraints_str = ',\n'.join([f"{('Satisfied' if mofn.satisfied else 'Not Satisfied')} {mofn.m}-of-\n{[condition for condition in mofn.conditions]}" for mofn in node.hard_constraints])
            label = f"Leaf\n(Class: {node.label},\nConstraints:\n{constraints_str})"
        else:
            label = f"Node\n(Split: {node.split.m}-of-\n{[condition for condition in node.split.conditions]},\nGain Ratio: {node.split.gain_ratio:.2f})"

        graph.node(node_id, label=label)

        if parent is not None:
            graph.edge(f"{id(parent)}", node_id)

        for child in node.children:
            self.to_graphviz(child, graph, node)

        return graph
