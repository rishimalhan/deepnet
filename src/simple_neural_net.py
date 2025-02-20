#! usr/bin/python3

"""
Doesn't assume structure (spatial, temporal, or otherwise)
Artificial neural network is a graph of nodes and edges
Neuron is a node in the neural network
Node with multiple inputs with weights associated and itself has a bias associated
sigma(xi * wi + b) where xi is the input, wi is the weight, b is the bias, and sigma is the activation function
Training neural network is tuning the weights and biases to achieve desired input output relationship
Neuron captures some non-linear representation using activation function without which everything will be linear

Loss Functions:
- Measure of how well the model is performing. Feedback mechanism to adjust the weights and biases.
- Estimate the difference between predicted output and expected output
- Common loss functions:
    - Mean Squared Error (MSE)
    - Cross-Entropy Loss

Cross entropy loss is useful to predict a class in multi-class classification problems

Optimizers:
- Find the best solution given a set of solutions and some constraints
- Common optimizers:
    - Stochastic Gradient Descent (SGD)
    - Adam
    - Adagrad
    - RMSprop
    - Adadelta
Calculate loss gradient from loss score wrt model weights and biases
- Update weights and biases in the direction of the negative gradient
- Gradient calculation is done using backpropagation
- Learning rate prescribes step size. Balance between time and accuracy.
- Best learning rate is deduced iteratively through optimization
- Error surface or manifold gives the landscape of the loss function
"""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import VisionDataset
import multiprocessing
from torch import nn
from torch.autograd import Variable
from IPython import embed

multiprocessing.set_start_method("fork")

DEVICE = "mps" if torch.mps.is_available() else "cpu"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


class IrisDataset(VisionDataset):
    def __init__(self, root, csv_file, transform=None, target_transform=None):
        """
        Initialize the Iris dataset by extending VisionDataset.
        Args:
            root (str): Root directory of the dataset.
            csv_file (str): Path to the CSV file.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on the target.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        # Read the CSV file and skip the header row
        self.data = pd.read_csv(csv_file, header=0)

        # Separate features and labels (assuming the label is the last column)
        self.features = self.data.iloc[:, :-1].values
        self.labels = self.data.iloc[:, -1].values

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            tuple: (features, label) where features is a tensor and label is a tensor.
        """
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        return features, label


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


iris_dataset = IrisDataset(
    root=os.path.join(ROOT, "data"),
    csv_file=os.path.join(ROOT, "data", "iris.csv"),
    transform=None,
)


class IrisDataLoader(DataLoader):
    def __init__(
        self, dataset, batch_size=16, shuffle=True, num_workers=0, train_split=0.8
    ):
        """
        Initialize the DataLoader for the Iris dataset with training and test splits.
        Args:
            dataset (Dataset): The IrisDataset instance.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data at every epoch.
            num_workers (int): Number of subprocesses to use for data loading.
            train_split (float): Proportion of data to use for training.
        """
        self.train_size = int(len(dataset) * train_split)
        self.test_size = len(dataset) - self.train_size
        self.train_dataset, self.test_dataset = random_split(
            dataset, [self.train_size, self.test_size]
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    def get_train_loader(self):
        """
        Return the DataLoader instance for training data.
        """
        return self.train_loader

    def get_test_loader(self):
        """
        Return the DataLoader instance for test data.
        """
        return self.test_loader


if __name__ == "__main__":
    dataloader = IrisDataLoader(
        dataset=iris_dataset, batch_size=30, shuffle=True, num_workers=10
    )
    model = SimpleNN(
        input_size=4, hidden_size1=100, hidden_size2=50, num_classes=3
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.9, dampening=0.0)

    for iter in range(100):
        train_loss = []
        test_loss = []
        for parameters, labels in dataloader.get_train_loader():
            model.train()
            parameters = Variable(data=parameters.to(DEVICE), requires_grad=True)
            labels = Variable(data=labels.to(DEVICE), requires_grad=False)
            parameters.retain_grad()
            # Forward pass
            outputs = model(parameters)
            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()  # Backward pass
            # Update weights
            optimizer.zero_grad()
            optimizer.step()
            # training accuracy
            training_accuracy = (outputs.argmax(dim=1) == labels).float().mean()
            train_loss.append(loss.item())
        model.eval()

        # Evaluate loss and accuracy on test
        with torch.no_grad():
            correct = 0
            total = 0
            for parameters, labels in dataloader.get_test_loader():
                parameters = Variable(data=parameters.to(DEVICE), requires_grad=False)
                labels = Variable(data=labels.to(DEVICE), requires_grad=False)
                outputs = model(parameters)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss.append(loss.item())
            test_accuracy = correct / total
        
        train_loss = torch.tensor(train_loss)
        test_loss = torch.tensor(test_loss)
        print(f"Iter {iter+1}, Training Loss: {train_loss.float().mean():.4f}, Training Accuracy: {training_accuracy:.4f}, Test Loss: {test_loss.float().mean():.4f}, Test Accuracy: {test_accuracy:.4f}")