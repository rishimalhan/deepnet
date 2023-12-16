#! /usr/bin/python3

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import numpy as np

with open("../data/dataset.pickle", "rb") as file:
    dataset = pickle.load(file)

# # Train a Random Forest classifier
# random_forest = RandomForestClassifier(n_estimators=500)
# random_forest.fit(dataset["x_train"], dataset["y_train"])

# # Predict with the Random Forest model
# predicted = random_forest.predict(dataset["x_test"])
# correct = np.sum(np.equal(predicted, dataset["y_test"]))
# accuracy = 100.0 * (correct / dataset["y_test"].shape[0])
# print(accuracy)

model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=3)
model.fit(dataset["x_train"], dataset["y_train"])
# Make predictions on the test set
predicted = model.predict(dataset["x_test"])
correct = np.sum(np.equal(predicted, dataset["y_test"]))
accuracy = 100.0 * (correct / dataset["y_test"].shape[0])
print(accuracy)
