import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from classes import *

iris = load_iris()
data = pd.DataFrame(data=np.column_stack((iris.data, iris.target)), columns=iris.feature_names + ['target'])

# Separate the input features and target labels
X = data.drop(columns=['target'])
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

oracle = Oracle(X_train, y_train)

# Constraints format: [(feature_index, value), ...]
constraints = [(0, 1), (2, 0)]

# Generate 10 new instances based on the model data distribution and the constraints
new_instances = oracle.generate_instances(constraints, 10)
print(new_instances)