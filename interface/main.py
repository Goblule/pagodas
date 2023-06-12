
import numpy as np

from ml.data import load_raw_fasta_file, load_raw_train_terms, clean_raw_fasta_df, get_preproc_data, save_preproc_data
from ml.preprocessing import encoding_target
from params import *
from pathlib import Path
from google.cloud import storage

# obo_file = Path(RAW_DATA_DIR).joinpath("go-basic.obo")
# graph_go, dict_go = read_obo_file(obo_file)
# print(f'\n✅ Graph from OBO file loaded')

def preprocess() -> None:

    """
    - loading raw data
    - embedding features
    - vectorizing target
    """

    # X_train for the moment are the features embedded by Sergei (cleaned of duplicates)
    X_train_filename = f'X_train_{NUM_OF_FEATS}_feats.csv'

    # X_ids are stored locally in a csv file (cleaned of duplicates)
    X_train_ids_filename = 'X_train_ids.csv'

    # y_train are stored locally in a csv file labelled with NUM_OF_LABELS
    y_train_filename = f'y_train_{NUM_OF_LABELS}_labels.csv'

    # y_labels are stored locally in a csv file labelled with NUM_OF_LABELS
    y_labels_filename = f'y_labels_{NUM_OF_LABELS}.csv'

    if STORAGE_DATA_KEY == 'local':
        # Define cache paths
        X_train_cache_path = Path(PREPROC_DATA_DIR).joinpath(X_train_filename)
        X_train_ids_cache_path = Path(PREPROC_DATA_DIR).joinpath(X_train_ids_filename)
        y_train_cache_path = Path(PREPROC_DATA_DIR).joinpath(y_train_filename)
        y_labels_cache_path = Path(PREPROC_DATA_DIR).joinpath(y_labels_filename)
        # X_train
        # Check if files exist
        if (X_train_cache_path.is_file() and X_train_ids_cache_path.is_file()):
            X_train = get_preproc_data(X_train_filename)
            X_train_ids = get_preproc_data(X_train_ids_filename)
        else:
            train_seq = load_raw_fasta_file()
            print(f'\n✅ Raw train sequences loaded')
            print(f'--- Train sequences with shape {train_seq.shape} ---')
            # Clean train_seq
            train_seq = clean_raw_fasta_df(train_seq)
            print(f'\n✅ Train sequences cleaned')
            print(f'--- Train sequences have now shape {train_seq.shape} ---')
            X_train_ids = train_seq.ids
            save_preproc_data(X_train_ids,X_train_ids_filename)
            # X_train = < TO IMPLEMENT > EMBEDDING FUNCTION
            # save_preproc_data(X_train,X_train_filename)
            pass
        # y_train
        # Check if files exist
        if (y_train_cache_path.is_file() and y_labels_cache_path.is_file()) :
            y_train = get_preproc_data(y_train_filename)
            y_labels = get_preproc_data(y_labels_filename)
        else:
            print(f'\n{y_train_filename} and {y_labels_filename} do not exist locally')
            print(f'\nPreprocessing raw data ...')
            # Load raw data
            train_terms = load_raw_train_terms()
            print(f'\n✅ Raw train terms loaded')
            print(f'--- Train terms with shape {train_terms.shape} ---')
            # Preproc target --> y_train, y_labels
            y_train, y_labels = encoding_target(train_terms,X_train_ids.ids)
            print(f'\n✅ Preprocessing done')
            print(f'--- Train target shape {y_train.shape} ---')
            print(f'--- Encoding labels shape {y_labels.shape} ---')
            print(f'\n✅ Saving results in local cache')
            # Save y_train, y_labels
            save_preproc_data(y_train,y_train_filename)
            save_preproc_data(y_labels,y_labels_filename)

    if STORAGE_DATA_KEY == 'gcs':
        # Initialize client
        client = storage.Client()
        # Get bucket
        bucket = client.get_bucket(BUCKET_NAME)
        # X_train
        # Check if files exist in the bucket
        if (storage.Blob(bucket=bucket,name=f'preproc_data/{X_train_filename}').exists(client) and \
            storage.Blob(bucket=bucket,name=f'preproc_data/{X_train_ids_filename}').exists(client)) :
            X_train = get_preproc_data(X_train_filename)
            X_train_ids = get_preproc_data(X_train_ids_filename)
        else:
            train_seq = load_raw_fasta_file()
            print(f'\n✅ Raw train sequences loaded')
            print(f'--- Train sequences with shape {train_seq.shape} ---')
            # Clean train_seq
            train_seq = clean_raw_fasta_df(train_seq)
            print(f'\n✅ Train sequences cleaned')
            print(f'--- Train sequences have now shape {train_seq.shape} ---')
            X_train_ids = train_seq.ids
            save_preproc_data(X_train_ids,X_train_ids_filename)
            # X_train = < TO IMPLEMENT > EMBEDDING FUNCTION
            # save_preproc_data(X_train,X_train_filename)
            pass
        # y_train
        # Check if files exist in the bucket
        if (storage.Blob(bucket=bucket,name=f'preproc_data/{y_train_filename}').exists(client) and \
            storage.Blob(bucket=bucket,name=f'preproc_data/{y_labels_filename}').exists(client)) :
            y_train = get_preproc_data(y_train_filename)
            y_labels = get_preproc_data(y_labels_filename)
        else:
            print(f'\n{y_train_filename} and {y_labels_filename} do not exist locally')
            print(f'\nPreprocessing raw data ...')
            # Load raw data
            train_terms = load_raw_train_terms()
            print(f'\n✅ Raw train terms loaded')
            print(f'--- Train terms with shape {train_terms.shape} ---')
            # Preproc target --> y_train, y_labels
            y_train, y_labels = encoding_target(train_terms,X_train_ids.ids)
            print(f'\n✅ Preprocessing done')
            print(f'--- Train target shape {y_train.shape} ---')
            print(f'--- Encoding labels shape {y_labels.shape} ---')
            print(f'\n✅ Saving results in gcs cloud')
            # Save y_train, y_labels
            save_preproc_data(y_train,y_train_filename)
            save_preproc_data(y_labels,y_labels_filename)


if __name__ == '__main__':
    preprocess()
