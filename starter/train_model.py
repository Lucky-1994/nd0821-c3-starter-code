# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.

import os
import pathlib
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import build_metrics_string, compute_model_metrics, compute_model_metrics_on_slices, \
    inference, save_pickle, train_model

# Add code to load in the data.

data_path = os.path.join(pathlib.Path(__file__).parent.parent, "data", "updated_census.csv")
data = pd.read_csv(data_path)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

predictions_train = inference(model, X_train)
precision_train, recall_train, fbeta_train = compute_model_metrics(y_train, predictions_train)
print(build_metrics_string(precision_train, recall_train, fbeta_train, prefix="Train: \n\t"))

predictions_test = inference(model, X_test)
precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, predictions_test)
print(build_metrics_string(precision_test, recall_test, fbeta_test, prefix="Test: \n\t"))

compute_model_metrics_on_slices(model, test, cat_features, encoder, lb)

model_folder_path = os.path.join(pathlib.Path(__file__).parent.parent, "model")

model_path = os.path.join(model_folder_path, "model.pkl")
encoder_path = os.path.join(model_folder_path, "encoder.pkl")
lb_path = os.path.join(model_folder_path, "lb.pkl")

save_pickle(model_path, model)
save_pickle(encoder_path, encoder)
save_pickle(lb_path, lb)
