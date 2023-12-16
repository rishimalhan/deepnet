#! /usr/bin/python3

"""
A good start point to Machine Learning is logistic regression
Very basic form of ML has been applying statistical models to data analysis
Applying Baye's theorem to the data assuming inputs are independent is Naive Baye's algorithm
Logistic Regression or logreg is based on this principle

https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be

Used to classify two linearly separable groups. This assumption makes binary logreg fast.
"""

import pickle
import tqdm
import numpy as np
import torch
from torch.nn.functional import normalize
import logging

logging.basicConfig()
logger = logging.getLogger("logisitic_regression")
logger.setLevel(logging.INFO)


class LogReg(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension):
        self._ip_dim = input_dimension
        self._op_dim = output_dimension
        self._epochs = 100000
        self._learning_rate = 0.001
        super(LogReg, self).__init__()
        self._create_model()
        logger.info(
            f"Initialized linear model: {self.linear_model} of type: {type(self.linear_model)}"
        )

    def _create_model(self):
        self.linear_model = torch.nn.Linear(self._ip_dim, self._op_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear_model(x))


def compute_stats(model, criterion, x, y):
    with torch.no_grad():
        outputs = torch.squeeze(model(x))
        loss = criterion(outputs, y).item()
        predicted = outputs.round().detach().numpy().astype(np.int8)
        expected = y.detach().numpy().astype(np.int8)
        correct = np.sum(np.equal(predicted, expected))
        accuracy = 100.0 * (correct / y.size(0))
    return accuracy, loss


def convert_element(element):
    if element == "+":
        return 1.0
    elif element == "-":
        return 0.0
    try:
        float(element)
        return float(element)  # Keep numbers unchanged
    except:
        number = 0
        for elem in element:
            number += ord(elem)
        return float(number)


try:
    with open("../data/dataset.pickle", "rb") as file:
        dataset = pickle.load(file)
except Exception as exc:
    logger.warning(f"Couldn't load dataset pickle due to: {exc}")
    credit_approvals = np.loadtxt(
        "../data/credit_approvals.csv", delimiter=",", dtype=str
    )
    converted_data = np.vectorize(convert_element)(credit_approvals)
    dataset = {}
    dataset["x_train"] = converted_data[:500, :-1]
    dataset["x_test"] = converted_data[500:, :-1]
    dataset["y_train"] = converted_data[:500, -1]
    dataset["y_test"] = converted_data[500:, -1]
    with open("../data/dataset.pickle", "wb") as file:
        pickle.dump(dataset, file)

x_train = dataset.get("x_train")
y_train = dataset.get("y_train")
x_test = dataset.get("x_test")
y_test = dataset.get("y_test")

model = LogReg(x_train.shape[1], 1)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=model._learning_rate)
x_train, x_test = normalize(torch.Tensor(x_train).float(), dim=1), normalize(
    torch.Tensor(x_test).float(), dim=1
)
y_train, y_test = torch.Tensor(y_train).float(), torch.Tensor(y_test).float()

iter = 0
for epoch in tqdm.tqdm(range(model._epochs), desc="Training Epochs"):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(torch.squeeze(outputs), y_train)
    loss.backward()
    optimizer.step()

    iter += 1
    if iter % int(model._epochs * 0.1) == 0:
        (accuracy, loss) = compute_stats(model, criterion, x_train, y_train)
        logger.info(
            f"Accuracy: {accuracy} and Loss: {loss} for epoch: {epoch} for training data"
        )
        (accuracy, loss) = compute_stats(model, criterion, x_test, y_test)
        logger.info(
            f"Accuracy: {accuracy} and Loss: {loss} for epoch: {epoch} for testing data"
        )

with open("../model/cats_vs_dogs.pickle", "wb") as file:
    pickle.dump(model, file)
