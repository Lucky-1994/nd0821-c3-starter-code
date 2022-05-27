import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier

from starter.ml.model import compute_model_metrics, inference, train_model


@pytest.fixture(scope="module")
def training_data():
    training_data = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
        [1, 1, 0],
    ])
    return training_data


@pytest.fixture(scope="module")
def labels():
    labels = np.array([0, 1, 0, 0, 1])
    return labels


@pytest.fixture(scope="module")
def trained_model(training_data, labels):
    trained_model = RandomForestClassifier().fit(training_data, labels)
    return trained_model


@pytest.fixture(scope="module")
def predictions():
    predictions = np.array([0, 1, 0, 1, 1])
    return predictions


def test_train_model(training_data, labels):
    model = train_model(training_data, labels)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics(labels, predictions):
    precision, recall, fbeta = compute_model_metrics(labels, predictions)
    assert 2 / 3 == precision
    assert 1.0 == recall
    assert 0.8 == fbeta


def test_inference(trained_model, training_data):
    predictions = inference(trained_model, training_data)
    assert [0, 1, 0, 0, 1] == predictions.tolist()
