# Put the code for your API here.
import os
import pathlib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Instantiate the app.
from starter.ml.data import process_data
from starter.ml.model import inference, load_model

app = FastAPI()

model_path = os.path.join(pathlib.Path(__file__).parent, "model", "model.pkl")
model, encoder, lb = load_model(model_path)

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
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


@app.get("/")
async def say_hello():
    return {"greeting": "Hello!"}


@app.post("/")
async def predict(features: Features):
    df = pd.DataFrame([features.dict(by_alias=True)])
    x, _, _, _ = process_data(df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)
    prediction = inference(model, x).tolist()[0]
    return {"prediction": prediction}
