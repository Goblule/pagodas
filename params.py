import os

# PREPROC
NUM_OF_LABELS = int(os.environ.get('NUM_OF_LABELS')) #size of the encoded target
NUM_OF_FEATS = int(os.environ.get('NUM_OF_FEATS')) #size of the embedding

# GCP PROJECT
GCP_PROJECT = os.environ.get('GCP_PROJECT') #name of the gcp project
GCP_REGION = os.environ.get('GCP_REGION')

# STORAGE DATA
STORAGE_DATA_KEY = os.environ.get('STORAGE_DATA_KEY') # local, gcs

# GCP CLOUD STORAGE
BUCKET_NAME = os.environ.get('BUCKET_NAME')

# DATA DIRS
RAW_DATA_DIR = os.path.join('.', 'raw_data', 'Train')
PREPROC_DATA_DIR = os.path.join('.', 'preproc_data')
MODEL_DATA_DIR = os.path.join('.', 'models')
