#! /usr/bin/python3

"""
Ability to have a high level understanding from images and videos
Do tasks that human visual system can do
Single objects:
- Classicfication and localization
Multiple objects:
- Object detection
- Segmentation

Examples:
- Apple's face ID (true depth camera). They built specialized hardware for this. 30000 dots
- Self-driving cars
- Medical imaging

Understanding images:
- Pixelated images are represented as a matrix of numbers
- RGB channel is three images or arrays of numbers
- Gray scale is either 0 or a value greater than 0
- It is important to note that each pixel's value is relative to neighbors or surrounding pixels
- In previous example the petal height, width, and other data was easily understood
- However, an image is a huge array of pixel unlike the previous example where it is more structured
- To capture all pixels together similar to linear neural network would require a huge number of weights
- The scale of weights would not even fit within the memory of the computer

CNNs:
- Use convolutions instead of building the entire neural network for individual weights
- Convolution -> Activation -> Pooling -> Fully connected layer
- Input activation map -> Kernel -> Output activation map
- Kernel has as many channels as the input activation map
- Kernel captures features like edges, corners, and other patterns
- Weights of the kernel are learnt during training to capture features
- RGB -> 3 kernels -> output -> Add outputs to single activation output
- Now keep in mind that the above is for one kernel only
- If we have multiple kernels then the output activation map will have same number of channels
- In short kernal has same number of channels as the input activation map and outputs one activation map
- Kernel has height, width, input channels, and output channels

Stride, Padding, and Pooling:
Stride represents by how many pixels to move the kernel
Along row dimension: output_size = (input_size + 2*padding - kernel size) / stride + 1
Pooling has a kernel size and stride. It reduces the size of the activation map.
Max pooling takes the maximum value from the kernel size
Average pooling takes the average value from the kernel size


Normalize the data:
- Prevents vanishing gradients i.e. the gradients become too small to update the weights at the beginning of the network
- Exploding gradients is the opposite of vanishing gradients
- Gradients act uniformly for the network

Regularization:
Dropout is a regularization technique where a random set of neurons are dropped out during training
- This prevents overfitting
- Typically it is 50% of the neurons
- So live neurons need to step in to take care of learning

Implementation reference:
loader = DataLoader(train_data, batch_size=10000, shuffle=False)
images, _ = next(iter(loader))  # Get all images in a batch
mean = images.mean().item()
std = images.std().item()
"""

import os
import numpy as np
import torch
import multiprocessing
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.sgd import SGD
from torch.autograd import Variable

multiprocessing.set_start_method("fork")

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEVICE = "mps" if torch.mps.is_available() else "cpu"
EPOCHS = 100

tf_composed = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )
train_data = MNIST(
    root=os.path.join(ROOT, "data"),
    train=True,
    download=True,
    transform=tf_composed,
)
test_data = MNIST(
    root=os.path.join(ROOT, "data"),
    train=False,
    download=True,
    transform=tf_composed,
)

mnist_train_loader = DataLoader(
    train_data, shuffle=True, batch_size=1024, num_workers=2, pin_memory=True
)
mnist_test_loader = DataLoader(
    test_data, shuffle=True, batch_size=1024, num_workers=2, pin_memory=True
)

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        # Initial size of image: 28x28x1 since it is MNIST dataset
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        # 24x24x20
        self.relu1 = nn.ReLU()
        # Same as above
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=5)
        # 20x20x30
        self.dropout = nn.Dropout2d(p=0.5)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # 10x10x30
        self.relu2 = nn.ReLU()
        self.linear1 = nn.Linear(in_features=3000, out_features=500)
        self.linear2 = nn.Linear(in_features=500, out_features=10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.relu2(x)
        x = x.view(-1, 3000)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
# Training loop
model = ConvNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.9, dampening=0)

for epoch in range(EPOCHS):
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []
    model.train()
    for i, (images, labels) in enumerate(mnist_train_loader):
        images = Variable(images).to(DEVICE)
        labels = Variable(labels).to(DEVICE)
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == labels).sum().item()
        train_loss.append(loss.item())
        train_accuracy.append(correct / labels.size(0))

    # validation
    model.eval()
    for i, (images, labels) in enumerate(mnist_test_loader):
        images = Variable(images).to(DEVICE)
        labels = Variable(labels).to(DEVICE)
        output = model(images)
        loss = criterion(output, labels)
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == labels).sum().item()
        test_loss.append(loss.item())
        test_accuracy.append(correct / labels.size(0))

    print(f"Epoch: {epoch}, Train Loss: {np.average(train_loss):.2f}, Train Accuracy: {np.average(train_accuracy):.2f}, " + \
    f"Test Loss: {np.average(test_loss):.2f}, Test Accuracy: {np.average(test_accuracy):.2f}")