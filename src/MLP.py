import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from datetime import datetime


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.MLP = nn.Sequential(
            nn.Linear(10000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 5)
        )

    def forward(self, x):
        x = self.MLP(x)
        return x
    
    def predict(self, X):
        self.to('cpu')
        samples = torch.from_numpy(X).float()
        with torch.no_grad():
            outputs = self(samples)
            predicted = torch.argmax(outputs.data, axis=1)

        return predicted
    
    def fit(self, train_dataset, batch_size=128, num_epochs=5, PATH=None, device='cpu'):
        # Multi-layer Perceptron classifier
        criterion = CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=0.001)
        trainloader = DataLoader(train_dataset, batch_size=batch_size)
        losses = []
        
        running_loss = 0.0
        for epoch in range(num_epochs):   
            for i, (inputs, labels) in enumerate(trainloader, start=0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if epoch % 1 == 0 and i==0:    # print every epoch
                    print(f'[{epoch+1}] loss: {running_loss:.6f}')
                    losses.append((epoch, running_loss))
                    running_loss = 0.0

        if not PATH:
            t = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            PATH = f'models/MLP_{t}.pth'
        torch.save(self.state_dict(), PATH)

        return self
    