#!/usr/local/bin/python3

import os
import sys

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)

sys.path.append(ROOT)

DEVICE = "mps"
EPOCH = 2000

from torch import Tensor, float32, save, load
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn import (
    Module,
    Linear,
    CrossEntropyLoss,
    Sequential,
    ReLU,
    Dropout,
)
from torch.optim.adagrad import Adagrad
from src.utils import full_evaluation

activation_fn = ReLU


class CustomDataSet(Dataset):
    def __init__(self, file_name: str):
        abs_path_to_file = os.path.abspath(os.path.join(ROOT, "data", file_name))
        self._dataset_csv = np.loadtxt(
            fname=abs_path_to_file, delimiter=",", skiprows=1
        )
        self._normalize()

    def _normalize(self):
        self._dataset_csv[:, :-1] /= np.linalg.norm(self._dataset_csv[:, :-1], axis=0)

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._dataset_csv[index, :-1].astype(np.float32),
            self._dataset_csv[index, -1].astype(np.int32) - 1,  # Labels start from 0
        )

    def __len__(self) -> int:
        return self._dataset_csv.shape[0]

    def cols(self) -> int:
        return self._dataset_csv.shape[1]

    def labels(self) -> np.ndarray:
        return dataset._dataset_csv[:, -1] - 1


class ANN(Module):
    def __init__(
        self,
        input_dim: int = 4,
        num_classes: int = 3,
    ):
        super().__init__()
        self.sequence = Sequential(
            Linear(in_features=input_dim, out_features=50, device=DEVICE),
            ReLU(),
            Linear(in_features=50, out_features=num_classes, device=DEVICE),
        )

    def forward(self, x) -> Tensor:
        out = self.sequence.forward(x)
        return out


if __name__ == "__main__":
    EVAL = True

    # sepal_length,sepal_width,petal_length,petal_width,species
    # "Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2
    dataset = CustomDataSet("iris.csv")
    dataloader = DataLoader(
        dataset=dataset, batch_size=150, shuffle=True, pin_memory=True
    )
    if not EVAL:
        model = ANN(input_dim=dataset.cols() - 1, num_classes=3)
        model.train(mode=True)
        criterion = CrossEntropyLoss()
        optimizer = Adagrad(params=model.parameters(), lr=0.01)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable_params}")
        accuracies = []
        for i in range(EPOCH):
            losses = []
            for x, labels in dataloader:
                optimizer.zero_grad()
                labels = labels.long().to(DEVICE)
                x = x.to(DEVICE)
                model.train()
                optimizer.zero_grad()
                outputs = model.forward(x)
                loss = criterion.forward(outputs, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            # Accuracy calculation on entire dataset
            ip = Tensor(dataset._dataset_csv[:, :-1]).type(dtype=float32).to(DEVICE)
            outputs = model.forward(ip)
            predictions = outputs.to("cpu").max(dim=1).indices.numpy()
            labels = dataset.labels()
            accuracy = np.sum(predictions == labels) / len(dataset) * 100.0
            accuracies.append(accuracy)
            if i % 100 == 0:
                print(f"Epoch #{i+1}. Loss: {np.average(losses)}. Accuracy: {accuracy}")
            # if len(accuracies) > 30 and np.all(
            #     np.subtract(accuracies[-20:-2], accuracies[-19:-1]) < 1
            # ):
            #     print("No progress in accuracy. Exiting")
            #     break
        save(
            model,
            os.path.join(ROOT, "model", "petal.model.pth"),
        )
    else:
        model = load(os.path.join(ROOT, "model", "petal.model.pth"))

        # Get data
        X = dataset._dataset_csv[:, :-1]
        y = dataset.labels()

        # Run evaluation
        full_evaluation(model, X, y, device=DEVICE)
