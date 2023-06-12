#imports
import os
from pathlib import Path
from tensorflow.keras import models, layers, metrics
from tensorflow.keras.saving import load_model
from tensorflow.keras.callbacks import EarlyStopping
from skmultilearn.adapt import MLkNN
from google.cloud import storage
from params import *

#functions

def MlkNN(neighbors):
    #instanciate the KNN classifier
    classifier = MLkNN(k=neighbors)
    return classifier

def dense(n_layers:int,input_neurons:int):

  ''' Function that creates a dense model and returns it'''

  #Instanciate the sequential model
  model = models.Sequential()

  #Input layer
  model.add(layers.Dense(input_neurons,activation = 'relu',
                         input_dim=NUM_OF_FEATS))

  #Middle layers through loop
  for i in range(1,n_layers):
    model.add(layers.Dense(input_neurons/2**i,activation = 'relu'))

  #Output layer
  model.add(layers.Dense(NUM_OF_LABELS,activation='sigmoid'))

  #Compile the model
  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=[metrics.AUC(),metrics.Recall(),'accuracy'])

  print(f'''✅ Dense model initialized, with {n_layers} layers,
        {input_neurons} input neurons''')

  return model


def LSTM(input_units:int):
    ''' Function that creates a tensorflow RNN with LSTM layers and
    returns it'''

    #Instanciate the sequential model
    model_RNN = models.Sequential()

    #LSTM layers
    model_RNN.add(layers.LSTM(units=input_units,activation='tanh',
                              return_sequences=True,
                              input_shape=(NUM_OF_FEATS, 1)))
    model_RNN.add(layers.LSTM(units=128,
                              activation='tanh',
                              return_sequences=False))

    #Dense layer
    model_RNN.add(layers.Dense(units=3000,activation='relu'))

    #Output layer
    model_RNN.add(layers.Dense(NUM_OF_LABELS,activation='sigmoid'))

    #Compile the model
    model_RNN.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[metrics.AUC(),metrics.Recall(),'accuracy'])

    print(f'''✅ RNN model initialized, with 2 LSTM layers,
          {input_units} input units''')

    return model_RNN


def ResLSTM(input_units:int):

    '''Function that creates a tensorflow ResLSTM model and returns it'''

    # Input layer
    inputs = layers.Input(shape=(NUM_OF_FEATS, 1))

    # Residual block
    residual = inputs

    # LSTM layer
    x = layers.Bidirectional(layers.LSTM(input_units,
                                         return_sequences=False))(inputs)
    x = layers.Add()([x, residual])

    # Flatten layer
    x = layers.Flatten()(x)

    # Dense layers
    x = layers.Dense(3000, activation='relu')(x)
    outputs = layers.Dense(NUM_OF_LABELS, activation='sigmoid')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[metrics.AUC(),metrics.Recall(),'accuracy'])

    print(f'''✅ ResLSTM model initialized, with 1 LSTM layers,
          {input_units} input units''')

    return model


def CNN_LSTM(input_filters,kernel_size):

    '''Function that creates a tensorflow CNN/LSTM hybrid model and
    returns it'''

    # Instanciate sequential model
    model_CNN_LSTM = models.Sequential()

    # Conv1D layer for spatial pattern detection
    model_CNN_LSTM.add(layers.Conv1D(input_filters,
                                     kernel_size=kernel_size,
                                     input_shape = (NUM_OF_FEATS,1),
                                     padding='same',
                                     activation='relu'))
    # pooling layer
    model_CNN_LSTM.add(layers.MaxPooling1D(pool_size=2))

    # LSTM layer for sequence modeling
    model_CNN_LSTM.add(layers.LSTM(units=64,
                                   dropout=0.2,
                                   recurrent_dropout=0.2))

    # Dense layer for non-linear transformations
    model_CNN_LSTM.add(layers.Dense(1500, activation='relu'))

    # Output layer (fully connected layer for classification)
    model_CNN_LSTM.add(layers.Dense(NUM_OF_LABELS, activation='sigmoid'))

    # Compile the model
    model_CNN_LSTM.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=[metrics.AUC(),metrics.Recall(),'accuracy'])

    print(f'''✅ ResLSTM model initialized, with 1 Conv1D,
          {input_filters} input filters and 1 LSTM layer''')

    return model_CNN_LSTM


def save_model(model,model_name):

    '''Function that saves a model either in the local directory or in gcs
    cloud. The option is specified via the environment variable
    STORAGE_DATA_KEY (local, gcs)
    '''

    ##define the path
    cache_path = Path(MODEL_DATA_DIR).joinpath(model_name)
    #save the model to the path
    model.save(cache_path)
    print(f"✅ {model_name} saved locally to {MODEL_DATA_DIR}")

    if STORAGE_DATA_KEY == 'gcs':
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(f'models/{model_name}')
        blob.upload_from_filename(cache_path)
        print(f"✅ {model_name} saved to GCS {BUCKET_NAME} bucket")
    pass


def load_train_model(model_filename):

    '''Function that loads a model either from the local directory or from
    gcs cloud. The option is specified via the environment variable
    STORAGE_DATA_KEY (local, gcs).'''

    if STORAGE_MODEL_KEY == 'local':
        cache_path = Path(MODEL_DATA_DIR).joinpath(model_filename)
        print(f"\nLoading local model file {model_filename} ...")
        model = load_model(cache_path)

    if STORAGE_MODEL_KEY == 'gcs':
        # Path of model file
        model_file = f'models/{model_filename}'
        print(f"\nLoading from gcs cloud model file {model_filename} ...")
        # Initialize client
        client = storage.Client()
        # Get bucket
        bucket = client.get_bucket(BUCKET_NAME)
        # Get blob
        blob = bucket.get_blob(model_file)
        # Define path
        model_bucket_file = os.path.join(f'gs://{BUCKET_NAME}', blob.name)
        # Load model
        model = load_model(model_bucket_file)
        print(f"✅ {model_file} loaded from gcs")

    return model


def train_model(model,X_train,y_train,epochs,batch_size,
                validation_split,patience) -> tuple :

    '''Function that trains a given model and returns the trained model and
    the history'''

    #define the early stopping
    es = EarlyStopping(patience=patience,restore_best_weights=True)
    #fit the model
    history = model.fit(X_train,
              y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=validation_split,
              callbacks = [es],
              verbose=2)

    return model, history
