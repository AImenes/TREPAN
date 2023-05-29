from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from classes import *
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

#Train ANN to represent with TREPAN
iris = load_iris()
X = iris.data
y = iris.target
# Create a dictionary to map class IDs to class labels
id_to_class_dict = {i: name for i, name in enumerate(iris.target_names)}


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
print(f"The ANN has an accuracy of: {accuracy:.2f}")

conf_matrix_ann_training = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_ann_training, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted by ANN')
plt.ylabel('Actual test labels')
plt.title('Confusion Matrix for ANN model used as oracle.')
plt.show(block=False)

# TREPAN
oracle = Oracle(model, X_train, y_train)

#Parameters
max_number_of_nodes = 20
number_of_instances, max_conditions = X.shape
S_min = number_of_instances // 10
max_children_per_node = 5
proportion_to_determine_class_in_leaf_node = 0.75
minimum_how_many_instances_in_a_split_evaluation=20

trepan = TREPAN(oracle=oracle, X=X_train, y=y_train, max_tree_size=max_number_of_nodes, max_conditions=max_conditions, max_children=max_children_per_node, cutoff=proportion_to_determine_class_in_leaf_node, num_of_instances=minimum_how_many_instances_in_a_split_evaluation)
trepan.fit()
trepan.print_tree()
graph = trepan.to_graphviz(id_to_class_dict)
graph.render("trepan_tree", view=True, cleanup=True)
print("Image successfully generated and placed in the working directory.")

# Predictions
y_test_pred = trepan.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy Score of TREPAN to original test set:\t", accuracy)


# Confusion matrix
confusion_mat_trepan = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix for original test set:\t")
print(confusion_mat_trepan)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_mat_trepan, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted by TREPAN')
plt.ylabel('Actual test labels')
plt.title('Confusion Matrix for TREPAN prediction')
plt.show(block=False)

#Fidelity
accuracy = accuracy_score(y_pred, y_test_pred)
print("Fidelity Score to our Oracle model:\t", accuracy)

#Fidelity confusion matric
confusion_mat_fidelity = confusion_matrix(y_pred, y_test_pred)
print("Fidelity Confusion Matrix between ANN and TREPAN:\t")
print(confusion_mat_fidelity)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_mat_fidelity, annot=True, cmap='Blues', fmt='g')
plt.xlabel('TREPAN')
plt.ylabel('ANN Model')
plt.title('Fidelity confusion matrix for prediction comparison between ANN-model and TREPAN')
plt.show(block=False)
