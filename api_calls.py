import json
import requests


URL = 'https://demo-income-app.herokuapp.com/'
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


def post(url, data):
    r = requests.post(url, data=data)
    return r.status_code, r.text


def get(url):
    r = requests.get(url)
    return r.status_code, r.text


def inference_post(data):
    status_code, text = post(URL, data)
    print('Status code:', status_code)
    print('Result:', text)


def root_get():
    status_code, text = get(URL)
    print('Status code:', status_code)
    print('Result:', text)


def main():
    root_get()
    inference_post(json.dumps(positive_example))
    inference_post(json.dumps(negative_example))


if __name__ == "__main__":
    main()
