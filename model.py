import torch
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        if not type(X) == torch.Tensor:
            self.X = torch.from_numpy(X.astype(np.float32))
        else:
            self.X = X
        if not type(y) == torch.Tensor:
            self.y = torch.from_numpy(y.astype(np.float32))
        else:
            self.y = y
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

# Create network
class NeuralNetwork(nn.Module):
    """
    NeuralNetwork(
    (layer_1): Linear(in_features=2, out_features=10, bias=True)
    (layer_2): Linear(in_features=10, out_features=1, bias=True)
    )
    """
    def __init__(self, input_size, hidden_size, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_size, output_dim) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)
        self.sm = nn.Softmax()
    def forward(self, x): 
        print(x.shape)
        out = self.fc1(x)
        print(out.shape)
        out = self.relu(out)
        print(out.shape)
        out = self.fc2(out)
        print(out.shape)
        out = self.sm(out)
        print(out.shape)
        return out



def train_oracle(X_data,y_data,target_names,**model_parameters):
    #Split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=model_parameters['test_size'], random_state=model_parameters["random_state"])

    # Instantiate training and test data
    train_data = Data(X_train, y_train)
    train_dataloader = DataLoader(dataset=train_data, batch_size=model_parameters['batch_size'], shuffle=True)
    test_data = Data(X_test, y_test)
    test_dataloader = DataLoader(dataset=test_data, batch_size=model_parameters['batch_size'], shuffle=True)

    model = NeuralNetwork(model_parameters['input_dim'], model_parameters['hidden_dim'], model_parameters['output_dim'])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_parameters['learning_rate'])
    loss_values = []

    for epoch in range(model_parameters['number_of_epochs']):
        for X, y in train_dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            pred = model(X)
            print(pred, y)
            loss = loss_fn(pred, y)
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()    
    print("Training complete.")

    # step = np.linspace(0, model_parameters['number_of_epochs'], len(loss_values))
    # fig, ax = plt.subplots(figsize=(8,5))
    # plt.plot(step, np.array(loss_values))
    # plt.title("Step-wise Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.show()



    # y_pred = []
    # y_true = []

    # # iterate over test data
    # for X, y in test_dataloader:
    #         output = model(X) # Feed Network

    #         output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    #         y_pred.extend(output) # Save Prediction
            
    #         labels = (torch.max(torch.exp(y), 1)[1]).data.cpu().numpy()
    #         y_true.extend(labels) # Save Truth



    # # Build confusion matrix
    # cf_matrix = confusion_matrix(y_true, y_pred)
    # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index = [i for i in target_names],
    #                     columns = [i for i in target_names])
    # plt.figure(figsize = (12,7))
    # sns.heatmap(df_cm, annot=True)
    # plt.savefig('output.png')
            
    return model