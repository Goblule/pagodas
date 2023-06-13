import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pagodas.ml.model import load_train_model
from pagodas.ml.preprocessing import get_embedding
import warnings
from pagodas.params import *

warnings.filterwarnings("ignore")
app = FastAPI()
app.state.trained_model = load_train_model(MODEL_PROD_NAME)
# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2

@app.get("/predict")
def predict(
        protein_sequence: str,  # 'MNSVTVSHAPYTITYHDDWEPVMSQLVEFYNEVASWLLRDETSPIPD'
    ):
    """
    Make a single protein prediction. Predicts the GO terms associated for this protein
    """

    #Embedd the protein sequence
    dico = {'seq' : [protein_sequence]}

    #create a dataframe for our protein
    X = pd.DataFrame(dico)

    #Embedd the protein sequence
    X_embedded = get_embedding(X)

    #initialize model
    model = app.state.model
    assert model is not None

    #predict
    sortie = model.predict(X_embedded)

    #return the protein functions
    return sortie


@app.get("/")
def root():
    return {'Greeting': 'Hello, and welcome to our protein API'}
