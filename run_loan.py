# from sklearn.datasets import load_iris
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from classes import *
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Train ANN to represent with TREPANs
loan_df = pd.read_csv("./data/loan.csv")
categorical = ['Gender','Married','Dependents','Education','Self_Employed', 'Credit_History','Property_Area']
continuous = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']


#TODO categorical with numbers?
loan_df.replace({'Gender':{'Male':1,'Female':0}},inplace=True)
loan_df.replace({'Married':{'Yes':1,'No':0}},inplace=True)
loan_df.replace({'Dependents':{'0':0,'1':1,'2':2,'3+':4}},inplace=True)
loan_df.replace({'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
loan_df.replace({'Self_Employed':{'Yes':1,'No':0}},inplace=True)
loan_df.replace({'Credit_History':{'1':1,'0':0}},inplace=True)
loan_df.replace({'Property_Area':{'Semiurban':0,'Urban':1,'Rural':2}},inplace=True)
loan_df.replace({'Loan_Status':{'Y':1,'N':0}},inplace=True)
#TODO, missing values
loan_df['Gender'].fillna(loan_df['Gender'].mode()[0],inplace=True)
loan_df['Married'].fillna(loan_df['Married'].mode()[0],inplace=True)
loan_df['Dependents'].fillna(loan_df['Dependents'].mode()[0],inplace=True)
loan_df['Education'].fillna(loan_df['Married'].mode()[0],inplace=True)
loan_df['Self_Employed'].fillna(loan_df['Self_Employed'].mode()[0],inplace=True)
loan_df['Credit_History'].fillna(loan_df['Credit_History'].mode()[0],inplace=True)

loan_df['Loan_Amount_Term'].fillna(loan_df['Loan_Amount_Term'].mean(),inplace=True)
loan_df['LoanAmount'].fillna(loan_df['LoanAmount'].mean(),inplace=True)

#data cleaning, outliers and duplicates
loan_df = loan_df.drop(['Loan_ID'], axis = 1)
# index_to_drop=loan_df[(loan_df['ApplicantIncome']>30000) | (loan_df['CoapplicantIncome']>=8000)].index
# loan_df.drop(index=index_to_drop,inplace=True,axis=0)
# loan_df.reset_index(inplace=True,drop=True)
# loan_df.drop_duplicates()

X = loan_df.drop("Loan_Status", axis=1)
# X = X.drop(columns=continuous)
categorical_features_idxs = [X.columns.get_loc(x) for x in categorical]
# continuous_features_idxs = [heart_df.columns.get_loc(x) for x in continuous]

#min max scaler only on continous features
sc = MinMaxScaler()
X[continuous] = sc.fit_transform(X[continuous])

# categorical_features_idxs = []
y = loan_df['Loan_Status']
print(X.head())

#TODO oversampling?
# X, y = SMOTE().fit_resample(X, y)

# Create a dictionary to map class IDs to class labels
# id_to_class_dict = {i: name for i, name in enumerate(iris.target_names)}
id_to_class_dict = {1 : 'granted', 0 : 'not_granted'}

model_path = "loan_model.pkl"
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# print(type(X_train))
# print(X_train[:5])
input_dimension = X_train.shape[1]
output_dimension = len(set(y_train))

# Check if the model already exists
if os.path.exists(model_path):
    # Load the existing model
    model = torch.load(model_path)
else:
    # Train a new model
    model = LoanNN(input_dimension, output_dimension)
    model.fit(X_train, y_train, epochs=20)

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

# exit(1)
# TREPAN
oracle = Oracle(model, X_train, y_train, categorical_features_idxs)

#Parameters
max_number_of_nodes = 10
number_of_instances, max_conditions = X.shape[0], (X.shape[1]//2)-2
# number_of_instances, max_conditions = X.shape[0], 5
S_min = number_of_instances // 10
max_children_per_node = 5
proportion_to_determine_class_in_leaf_node = 0.75
minimum_how_many_instances_in_a_split_evaluation=20

#TODO check to parametrize passing the data to TREPAN
trepan = TREPAN(oracle=oracle, X=X_train, y=y_train, max_tree_size=max_number_of_nodes, max_conditions=max_conditions, max_children=max_children_per_node, 
                cutoff=proportion_to_determine_class_in_leaf_node, num_of_instances=minimum_how_many_instances_in_a_split_evaluation,categorical_features_idxs=categorical_features_idxs)
trepan.fit()
trepan.print_tree()
graph = trepan.to_graphviz(id_to_class_dict)
graph.render("trepan_tree_loan", view=True, cleanup=True)
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
