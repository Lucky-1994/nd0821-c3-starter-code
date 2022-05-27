import json

from fastapi.testclient import TestClient

from main import app


def test_api_locally_get_root():
    client = TestClient(app)
    response = client.get("/")
    assert 200 == response.status_code
    assert {'greeting': 'Hello!'} == response.json()


def test_api_locally_post_positive_example():
    positive_example = {
        'age': 42,
        'workclass': 'Private',
        'fnlgt': 111483,
        'education': 'Bachelors',
        'education-num': 13,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Tech-support',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 50,
        'native-country': 'United-States',
    }
    client = TestClient(app)
    response = client.post("/", data=json.dumps(positive_example))
    assert 200 == response.status_code
    assert {'prediction': 1} == response.json()


def test_api_locally_post_negative_example():
    negative_example = {
        'age': 36,
        'workclass': 'Local-gov',
        'fnlgt': 103886,
        'education': 'Some-college',
        'education-num': 10,
        'marital-status': 'Divorced',
        'occupation': 'Handlers-cleaners',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 50,
        'native-country': 'United-States'
    }
    client = TestClient(app)
    response = client.post("/", data=json.dumps(negative_example))
    assert 200 == response.status_code
    assert {'prediction': 0} == response.json()

