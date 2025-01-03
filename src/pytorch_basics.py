#! /usr/bin/python3

"""
This script is recap from Deep learning with pytorch lecture from Anand Saha

Pytorch:
- Python platform for deep learning research and implementation
- Popularity due to two main reasons: 1) Dynamic computation graph 2) Numpy version for CUDA
- Dynamic computation graph or Imperative style
- Dynamic computation graph is a graph that is created on the fly as the operations are executed
- Imperative style is a style of programming where you write code that describes what you want to do, not how you want to do it
- Pytorch is a Imperative style of programming

Numpy and Pytorch:
- Pytorch is a numpy version for CUDA
- Shared memory between numpy and pytorch tensors
- Pytorch tensors can be converted to numpy arrays and vice versa

Pytorch tensors:
- Concept of scalar, vector, matrix, and then comes tensor
- Pytorch tensors are similar to numpy arrays
- Pytorch tensors are the basic building blocks of pytorch
- Pytorch tensors are multi-dimensional arrays
- Pytorch tensors are used to store data
- Dimensions are number of indexes required to access an element in a tensor
- Rank is the number of dimensions in a tensor

Pytorch Variable:
- We need to store parent of tensor, the gradient of the tensor, and the value of the tensor
# Example of creating a Variable
tensor = torch.Tensor([1.0, 2.0, 3.0])
variable = Variable(tensor, requires_grad=True)

Pytorch Datasets:
- Dataset is entire corpus of training and testing data
- Epoch is pass through entire dataset
- Iteration is pass through a batch (subset) of the dataset
- Data downloading, loading, and preprocessing utils are all part of torch.utils.data
- The package torchvision has most of the tools for handling data
- We can extend torch.utils.data.DataLoader class for custom data loading
- torchvision.transforms is used for manipulating the data with a sequence of transforms

Deep learning with Pytorch:
- Adding the ability for computers to see, hear, and understand the world like humans do
- Essence is to learn from massive amounts of data without feature engineering
- Neural net applies a sequence of spatial transformations to the input data
- Idea is to make simplified representations of the input data in each layer
- Hierarchy of representations then leads to a final output

- Computer vision:
    - Object detection (instance segmentation, semantic segmentation)
        - Instance segmentation segments each object into classes even if they belong to the same one
        - Semantic segmentation only classifies classes (would club objects of the same class)
    - Localization is finding the bounding box of the object or its position and orientation

- Sentiment analysis:
    - Sentiment analysis is a type of text classification
    - Sentiment analysis is the task of classifying text into a positive, negative, or neutral sentiment-based category

- Speech recognition:
    - Task of recognizing spoken words
    - Converting spoken words into text
"""

import multiprocessing
import os
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from IPython import embed

multiprocessing.set_start_method("fork")

DEVICE = "mps" if torch.mps.is_available() else "cpu"
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

one_tensor = Tensor().new_ones(size=(10, 10), device=DEVICE)
np_random_array = np.random.uniform(low=0.0, high=1.0, size=(10,))
random_tensor = torch.from_numpy(np_random_array)

x = Variable(data=random_tensor, requires_grad=True).view(10, 1)
y = 2 * torch.matmul(x.t(), x) + 3
y.retain_grad()
x.retain_grad()

tf_composed = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
)

trainset = CIFAR10(
    root=os.path.join(ROOT, "data"), train=True, transform=tf_composed, download=True
)
mnist_data = DataLoader(dataset=trainset, batch_size=32, shuffle=True, num_workers=10)

embed()
