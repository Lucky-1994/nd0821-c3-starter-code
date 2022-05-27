# Put the code for your API here.
import os
import pathlib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Instantiate the app.
from starter.ml.data import process_data
from starter.ml.model import inference, load_pickle

app = FastAPI()

model_folder_path = os.path.join(pathlib.Path(__file__).parent, "model")

model_path = os.path.join(model_folder_path, "model.pkl")
encoder_path = os.path.join(model_folder_path, "encoder.pkl")
lb_path = os.path.join(model_folder_path, "lb.pkl")

model = load_pickle(model_path)
encoder = load_pickle(encoder_path)
lb = load_pickle(lb_path)

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


class Features(BaseModel):
    age: int = Field(example=42)
    workclass: str = Field(example="Private")
    fnlgt: int = Field(example=111483)
    education: str = Field(example="Bachelors")
    education_num: int = Field(example=13, alias="education-num")
    marital_status: str = Field(example="Married-civ-spouse", alias="marital-status")
    occupation: str = Field(example="Tech-support")
    relationship: str = Field(example="Husband")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(example=0, alias="capital-gain")
    capital_loss: int = Field(example=0, alias="capital-loss")
    hours_per_week: int = Field(example=50, alias="hours-per-week")
    native_country: str = Field(example="United-States", alias="native-country")


@app.get("/")
async def say_hello():
    return {"greeting": "Hello!"}


@app.post("/")
async def predict(features: Features):
    df = pd.DataFrame([features.dict(by_alias=True)])
    x, _, _, _ = process_data(df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)
    prediction = inference(model, x).tolist()[0]
    if prediction == 0:
        prediction = "<= 50k"
    else:
        prediction = "> 50k"
    return {"Income": prediction}
