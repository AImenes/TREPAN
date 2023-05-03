from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from classes import *
import os
import torch


#Train ANN to represent with TREPAN
iris = load_iris()
X = iris.data
y = iris.target

model_path = "iris_model.pkl"
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if the model already exists
if os.path.exists(model_path):
    # Load the existing model
    model = torch.load(model_path)
else:
    # Train a new model
    model = IrisNN(4, 3)
    model.fit(X_train, y_train, epochs=30)

    # Save the trained model
    torch.save(model, model_path)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# TREPAN
oracle = Oracle(model, X_train, y_train)

#Parameters
max_number_of_nodes = 15
number_of_instances, max_conditions = X.shape
S_min = number_of_instances // 10
max_children_per_node = 5
proportion_to_determine_class_in_leaf_node = 0.65

trepan = TREPAN(oracle=oracle, X=X_train, y=y_train, max_tree_size=max_number_of_nodes, max_conditions=max_conditions, max_children=max_children_per_node, cutoff=proportion_to_determine_class_in_leaf_node, num_of_instances=number_of_instances)
trepan.fit()
trepan.print_tree()
graph = trepan.to_graphviz()
graph.render("trepan_tree", cleanup=True)
print("Image successfully generated")

# Predictions
y_test_pred = trepan.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy Score:", accuracy)

# Confusion matrix
confusion_mat = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(confusion_mat)

#Fidelity
accuracy = accuracy_score(y_pred, y_test_pred)
print("Fidelity Score:", accuracy)