import os

##################  VARIABLES  ##################
NUM_OF_LABELS = os.environ.get('NUM_OF_LABELS')
RAW_DATA_DIR = os.path.join('.', 'raw_data', 'Train')
PREPROC_DATA_DIR = os.path.join('.', 'preproc_data')
STORAGE_DATA_KEY = 'gcs'
BUCKET_NAME = 'project_pagodas'
