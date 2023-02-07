from sklearn.datasets import load_iris
import torch
from torch.nn.functional import one_hot

def load_dataset():
    iris = load_iris()
    y = torch.from_numpy(iris.target)
    y = one_hot(y, num_classes=-1)
    y = y.to(torch.float32)
    test = y.data[0]
    return iris.data, y, iris.feature_names, iris.target_names