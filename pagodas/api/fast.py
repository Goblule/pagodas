import pandas as pd
import numpy as np
import logging
import requests
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pagodas.ml.model import load_train_model
from pagodas.ml.preprocessing import get_embedding
from pagodas.ml.data import load_fasta_file
from pagodas.params import *

logging.getLogger('tensorflow').disabled = True

app = FastAPI()
app.state.trained_model = load_train_model(MODEL_PROD_NAME)
# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],  # Allows all origins
                   allow_credentials=True,
                   allow_methods=["*"],  # Allows all methods
                   allow_headers=["*"],  # Allows all headers
                   )

@app.get("/predict")
def predict(protein_sequence: str):
    """
    Make a single protein prediction. Predicts the GO terms associated for
    this protein
    i.e. MNSVTVSHAPYTITYHDDWEPVMSQLVEFYNEVASWLLRDETSPIPDKFFIQLKQPLRNKRVCVCGIDPY
         PKDGTGVPFESPNFTKKSIKEIASSISRLTGVIDYKGYNLNIIDGVIPWNYYLSCKLGETKSHAIYWDKI
         SKLLLQHITKHVSVLYCLGKTDFSNIRAKLESPVTTIVGYHPAARDRQFEKDRSFEIINVLLELDNKVPI
         NWAQGFIY
    """

    #Read the fasta with 142k proteins to get the headers and the sequences
    train_seq_df = load_fasta_file('train_sequences.fasta')
    print('loaded fasta file')

    #Path
    train_path = Path(PREPROC_DATA_DIR).joinpath('train_embeds.npy')
    train_ids_path = Path(PREPROC_DATA_DIR).joinpath('train_ids.npy')
    y_labels_path = Path(PREPROC_DATA_DIR).joinpath('y_labels_1500.npy')
    print('created paths')

    #Load the local embeddings
    X_train = np.load(train_path)
    X_train_ids = np.load(train_ids_path)

    #Load the encoded labels
    y_labels = np.load(y_labels_path,allow_pickle=True)

    print('loaded numpy arrays')

    #Create the embedding df
    ids = [x for x in X_train_ids] #create a list of ids
    embeds = [x for x in X_train] #create a list of embeds
    embedded_df = pd.DataFrame({'id':ids,'embed':embeds}) #create the dataframe
    print('created dataframe')

    #Merge the dataframe
    merged_df = pd.merge(train_seq_df, embedded_df, on='id')
    print('merged dataframe')

    #Change the index to the sequences
    merged_df.set_index('sequence',drop=True,inplace=True)
    print('changed index')

    #Get the embedding of the protein
    X_embedded = np.expand_dims(merged_df['embed'].get(protein_sequence),axis=0)
    print('got the embedding')

    '''if embedding_df:
        #If the embedding exists in 142k dataset we take it
        X_embedded = embedding_df['']

    else:
        #Embedd the protein sequence
        X_embedded = get_embedding(X)'''

    #initialize model
    model = app.state.trained_model
    assert model is not None
    print('initialized the model')

    #predict
    y_pred = model.predict(X_embedded)
    print('made a prediction')

    y_pred = np.reshape(y_pred, (1500))

    #create the final pred dataframe with the GO_terms
    df_pred = pd.DataFrame(y_pred,index=y_labels)
    df_pred.reset_index(inplace=True)
    df_pred.rename(columns={0:'Proba','index':'Function'},inplace=True)
    df_pred.sort_values(by=['Proba'],inplace=True,ascending=False)
    df_pred.reset_index(inplace=True,drop=True)
    df_pred = df_pred[df_pred['Proba']>0.3]
    print('organized the prediction')


    #return the protein functions
    return {key:round(float(value),4) for key,value in zip(df_pred['Function'],df_pred['Proba'])}


@app.get("/")
def root():
    return {'Greeting': 'Hello, and welcome to our protein API'}
