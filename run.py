from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from classes import *

iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = IrisNN(4,10,3)
model.fit(X, y, epochs=70)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

oracle = Oracle(model, X_train, y_train)

#Parameters
max_number_of_nodes = 30
number_of_instances, max_conditions = X.shape
max_children_per_node = 5
proportion_to_determine_class_in_leaf_node = 0.7

trepan = TREPAN(oracle=oracle, X=X_train, y=y_train, max_tree_size=max_number_of_nodes, max_conditions=max_conditions, max_children=max_children_per_node, cutoff=proportion_to_determine_class_in_leaf_node, num_of_instances=number_of_instances)
trepan.fit()
trepan.print_tree()
graph = trepan.to_graphviz()
graph.render("trepan_tree", cleanup=True)

# Predictions
y_test_pred = trepan.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy Score:", accuracy)

# Confusion matrix
confusion_mat = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(confusion_mat)