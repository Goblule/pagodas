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
  print(f"✅ Dense model initialized, with {n_layers} layers, {input_neurons} input neurons")

  #compile the model
  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=[metrics.AUC()])

  return model

def save_model(model,model_name):
    '''Function that saves the model'''
    ##define the path
    MODEL_DIR = os.path.join('drive/MyDrive/pagodas','models')
    model_file = Path(MODEL_DIR).joinpath(model_name)
    #save the model to the path
    model.save(model_file)
    pass

def load_model(model_file):
    '''Function that loads the model'''
    model = load_model(model_file, custom_objects=None, compile=True, safe_mode=True)
    return model

def train_model(model,X_train,y_train,epochs,batch_size,validation_split,patience):
    '''Function that trains the model'''
    #define the early stopping
    es = EarlyStopping(patience=patience,restore_best_weights=True)
    #fit the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,callbacks = [es],verbose=2)
    return model
