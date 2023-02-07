import torch
from classes.training_examples import TrainingExamples
# Query the oracle for all training data in which the oracle was trained upon. Get labels.

def predict(X, y, oracle):
    labels = list()
    X = torch.from_numpy(X)
    X = X.to(torch.float32)
    for x in X:
        predict = oracle(x)
        output = (torch.argmax(predict).item())
        #output = (torch.max(torch.exp(predict), 1)).data.cpu().numpy()
        labels.append(output)

    return TrainingExamples(X, y, labels)