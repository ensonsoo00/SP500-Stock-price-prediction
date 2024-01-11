
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score

class MatrixToFloatNN(nn.Module):
    
    def __init__(self,inputSize):
        super(MatrixToFloatNN, self).__init__()
        self.fc1 = nn.Linear(inputSize,int(inputSize/2)).float()
        self.fc2 = nn.Linear(int(inputSize/2), 1).float()
        #self.fc3 = nn.Linear(int(inputSize/4), 1).float()
        self.name = "Neural Network"

    def forward(self, x):
        #x = torch.flatten(x, 1)
        x = x.float()
        x = torch.relu(self.fc1(x)).float()
        x = self.fc2(x)
        return x


class NN:
    def __init__(self,X_train,X_test,y_train,y_test):
        self.name = "Neural Network"
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.inputSize = X_train[0].size
        self.outputSize = 1
        self.net = MatrixToFloatNN(self.inputSize).float()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(),lr=.1)
        self.r2 = 0
    def train(self):
        
        for epoch in range(5):
            for data in range(len(self.X_train)):
                self.optimizer.zero_grad()
                _data = torch.tensor(self.X_train[data],dtype=float)
                output = self.net(_data).float()
                target =  torch.tensor(self.y_train[data],dtype=float).float() # Example target float value
                self.loss = self.criterion(output, target)
                self.loss.backward()
                self.optimizer.step()

                if epoch % 10 == 0:
                    scaledLoss = self.loss.item()
                    print(f"Epoch: {epoch}, Loss: {scaledLoss}")
    def test(self):
        actual = []
        expected = []
        #
        for data in range(len(self.X_test)):
            self.optimizer.zero_grad()
            _data = torch.tensor(self.X_test[data],dtype=float).float()
            output = self.net(_data).float()
            target =  torch.tensor(self.y_test[data],dtype=float).float() # Example target float value
            loss = self.criterion(output, target)
            
            print(target.tolist(),output.tolist())
            #input()
            actual.append(target.tolist()[0])
            expected.append(output.tolist()[0])
        self.r2 = r2_score(expected,actual)
        return r2_score(expected,actual)
    def evaluate(self):
        return {"r2":self.r2,"model":self.net,"name":"Neural Network"}
    def error(self):
        return self.r2
def evaluation():
    print("Evaluating Neural Network...")