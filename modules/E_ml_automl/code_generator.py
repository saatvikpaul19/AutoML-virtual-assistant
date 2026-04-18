from __future__ import annotations

import json


def _notebook_from_source(title: str, source: str) -> str:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in source.splitlines()],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=2)


def generate_code_bundle(state: dict) -> dict:
    dataset = state.get("dataset") or "dataset"
    profile = state.get("dataset_profile", {})
    model = state.get("model") or "xgboost"
    lr = state.get("learning_rate", 0.001)
    batch = state.get("batch_size", 32)
    epochs = state.get("epochs_total", 10)
    layers = state.get("layers", 3)
    activation = state.get("activation", "relu")
    test_ratio = state.get("split_ratio", 0.2)
    preview_file = state.get("dataset_info", {}).get("preview_file", "path/to/your/data.csv")

    task_family = profile.get("task_family", "tabular_classification")

    if task_family == "image_classification":
        py_source = f'''import torch
import torch.nn as nn
import torch.optim as optim

# Dataset
dataset_name = "{dataset}"
learning_rate = {lr}
batch_size = {batch}
epochs = {epochs}
layers = {layers}
activation = "{activation}"
test_ratio = {test_ratio}

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        blocks = []
        in_channels = 1
        channels = [16, 32, 64, 128][:max(1, min(layers, 4))]
        for out_channels in channels:
            blocks += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3), keepdim=False)
        return self.classifier(x)

model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

print("Ready to train image model for", dataset_name)
'''
    elif task_family == "tabular_regression":
        py_source = f'''import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

dataset_name = "{dataset}"
learning_rate = {lr}
batch_size = {batch}
epochs = {epochs}
test_ratio = {test_ratio}
data_path = r"{preview_file}"

df = pd.read_csv(data_path)
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_ratio}, random_state=42)

model = XGBRegressor(
    learning_rate=learning_rate,
    n_estimators=epochs,
    max_depth={max(3, layers + 1)},
    verbosity=0,
)

model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print("RMSE:", rmse)
'''
    else:
        if model == "random_forest":
            model_code = f'''from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators={max(50, epochs * 10)}, random_state=42)'''
        elif model == "logistic_regression":
            model_code = f'''from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter={max(200, epochs * 20)})'''
        elif model == "mlp":
            model_code = f'''from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=({max(16, layers*16)},), learning_rate_init={lr}, batch_size={batch}, activation="{activation}", max_iter={epochs}, random_state=42)'''
        else:
            model_code = f'''from xgboost import XGBClassifier
model = XGBClassifier(
    learning_rate={lr},
    n_estimators={epochs},
    max_depth={max(3, layers + 1)},
    verbosity=0,
    eval_metric="logloss",
)'''

        py_source = f'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset_name = "{dataset}"
learning_rate = {lr}
batch_size = {batch}
epochs = {epochs}
layers = {layers}
activation = "{activation}"
test_ratio = {test_ratio}
data_path = r"{preview_file}"

df = pd.read_csv(data_path)
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_ratio}, random_state=42, stratify=y if y.nunique() < 30 else None
)

{model_code}

model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print("Accuracy:", acc)
'''

    return {
        "title": f"{dataset} experiment",
        "py_source": py_source,
        "ipynb_source": _notebook_from_source(f"{dataset} experiment", py_source),
    }