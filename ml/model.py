#imports
import os
from pathlib import Path
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, get_scorer_names
from tensorflow.keras import Model, models, layers, metrics
from tensorflow.keras.saving import load_model
from tensorflow.keras.callbacks import EarlyStopping
from google.cloud import storage
from params import *

#functions
def dense(n_layers:int,input_neurons:int,nlabels:int,nfeats:int):
  ''' Function that creates a dense tensorflow model'''
  #instanciate the sequential model
  model = models.Sequential()
  #input layer
  model.add(layers.Dense(input_neurons,activation = 'relu',input_dim=nfeats))
  #add the middle layers through loop
  for i in range(1,n_layers):
    model.add(layers.Dense(input_neurons/2**i,activation = 'relu'))
  #add the output layer
  model.add(layers.Dense(nlabels,activation='sigmoid'))
  print(f"✅ Dense model initialized, with {n_layers} layers, {input_neurons} input neurons")

  #compile the model
  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=[metrics.AUC()])

  return model

def LSTM(input_units:int,nlabels:int,nfeats:int):
    ''' Function that creates a tensorflow RNN with LSTM layers'''
    #instanciate sequential model
    model_RNN = models.Sequential()
    #lstm layers
    model_RNN.add(layers.LSTM(units=input_units,activation='tanh',return_sequences=True,input_shape=(nfeats, 1)))
    model_RNN.add(layers.LSTM(units=64,activation='tanh',return_sequences=False))
    #dense layer
    model_RNN.add(layers.Dense(units=64,activation='relu'))
    #output layer
    model_RNN.add(layers.Dense(nlabels,activation='sigmoid'))

    #compile the model
    model_RNN.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[metrics.AUC()])
    print(f"✅ RNN model initialized, with 2 LSTM layers, {input_units} input units")
    return model_RNN

def save_model(model,model_name):
    '''Function that saves the model'''
    ##define the path
    MODEL_LOCAL_DIR = f'models/{model_name}'
    #save the model to the path
    model.save(MODEL_LOCAL_DIR)
    print(f"✅ {model_name} saved locally to {MODEL_LOCAL_DIR}")
    if STORAGE_DATA_KEY == 'gcs':
        MODEL_BUCKET_PATH = os.path.join(f'gs://{BUCKET_NAME}', model_name)
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_BUCKET_PATH)
        blob.upload_from_filename(MODEL_BUCKET_PATH)
        model.save(MODEL_BUCKET_PATH)
        print(f"✅ {model_name} saved to GCS {BUCKET_NAME} bucket")
    pass

def load_model(model_file):
    '''Function that loads the model from gcs'''
    if STORAGE_DATA_KEY == 'gcs':
        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.get_blob(model_file)
        #print(blob)
        #model_file = 'models/dense_2L_256_1500_labels_baseline.h5'
        MODEL_LOAD_DIR = os.path.join(f'gs://{BUCKET_NAME}', blob.name)
        model = load_model(MODEL_LOAD_DIR)
        print(model.summary())
        print(f"✅ {model_file} loaded from gcs")

    return model

def train_model(model,X_train,y_train,epochs,batch_size,validation_split,patience):
    '''Function that trains the model'''
    #define the early stopping
    es = EarlyStopping(patience=patience,restore_best_weights=True)
    #fit the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,callbacks = [es],verbose=2)
    return model
