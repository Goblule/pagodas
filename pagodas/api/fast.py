import pandas as pd
import numpy as np
import requests
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pagodas.ml.model import load_train_model
from pagodas.ml.preprocessing import get_embedding
from pagodas.ml.data import load_raw_fasta_file, load_fasta_file
from pagodas.params import *

#warnings.filterwarnings("ignore")
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

    '''#Create dictionary with the protein sequence
    dico = {'seq' : [protein_sequence]}

    #Create a dataframe for our protein
    X = pd.DataFrame(dico)'''

    #Read the fasta with 142k proteins to get the headers and the sequences
    train_seq_df = load_fasta_file('train_sequences.fasta')
    print(train_seq_df.head())
    print(type(train_seq_df['id'][0]))

    print('loaded fasta file')

    #Path
    train_path = Path(PREPROC_DATA_DIR).joinpath('train_embeds.npy')
    train_ids_path = Path(PREPROC_DATA_DIR).joinpath('train_ids.npy')
    print('created paths')

    #Load the local embeddings
    X_train = np.load(train_path)
    X_train_ids = np.load(train_ids_path)
    print('loaded numpy arrays')
    print(X_train[0],X_train_ids[0])

    #Create the embedding df
    ids = [x for x in X_train_ids] #create a list of ids
    embeds = [x for x in X_train] #create a list of embeds
    embedded_df = pd.DataFrame({'id':ids,'embed':embeds}) #create the dataframe
    print('created dataframe')
    print(embedded_df.head())

    #Merge the dataframe
    merged_df = pd.merge(train_seq_df, embedded_df, on='id')
    print('merged dataframe')
    print(merged_df.head())

    #Change the index to the sequences
    merged_df.set_index('sequence',drop=True)
    print('changed index')
    print(merged_df.head())

    #Get the embedding of the protein
    X_embedded = merged_df['embed'][0]
    print('got the embedding')
    print(X_embedded)
    print(X_embedded.shape)

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
    sortie = model.predict(X_embedded)
    print('made a prediction')

    #return the protein functions
    return sortie


@app.get("/")
def root():
    return {'Greeting': 'Hello, and welcome to our protein API'}
