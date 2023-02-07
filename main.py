from trepan import trepan
from model import train_oracle
from dataset import load_dataset

# DATASET
X, y, feature_names, target_names = load_dataset()

# The ORACLE #
## Model Parameters
model_parameters = {
    "random_state":28,
    "test_size":0.2,
    "input_dim":len(feature_names),
    "hidden_dim":10,
    "output_dim":len(target_names),
    "batch_size":8,
    "learning_rate":0.01,
    "number_of_epochs":10}

## Initiate Oracle
oracle = train_oracle(X,y,target_names,**model_parameters)



# The algorithm, Trepan
## Parameters
trepan_parameters = {
  "epselon": 0.05,
  "delta": 0.05,
  "number_of_iternal_nodes": 10,
  "S_min": 10,
  "use_limit_on_iternal_nodes": True}

## Run Trepan
T = trepan(X, y, feature_names, oracle, **trepan_parameters)