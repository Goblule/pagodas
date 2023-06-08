#imports
from pathlib import Path
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, get_scorer_names
from tensorflow.keras import Model, models, layers, metrics
from tensorflow.keras.callbacks import EarlyStopping

#functions
def dense(n_layers:int,input_neurons:int,nlabels:int,nfeats:int):
  ''' Function that creates a dense tensorflow model'''
  #initialize the sequential model
  model = models.Sequential()
  #input layer
  model.add(layers.Dense(input_neurons,activation = 'relu',input_dim=nfeats))
  #add the middle layers through loop
  for i in range(1,n_layers):
    model.add(layers.Dense(input_neurons/2**i,activation = 'relu'))
  #add the output layer
  model.add(layers.Dense(nlabels,activation='sigmoid'))
  return model
