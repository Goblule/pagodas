
import numpy as np

from ml.data import load_raw_data_local, get_data_with_cache
from ml.preprocessing import encoding_target
from params import *
from pathlib import Path

def preprocess() -> None:

    """
    - loading raw data
    - embedding features
    - vectorizing target
    """

    train_terms, train_seq, graph_go, dict_go = load_raw_data_local(RAW_DATA_DIR)
    series_train_protein_ids = train_seq.ids

    print(f'✅ Raw Data loaded')
    print(f'--- Train terms with shape {train_terms.shape} ---')
    print(f'--- Train sequences with shape {train_seq.shape} ---')
    print(f'✅ Graph from OBO file loaded')

    # X_train for the moment are the features embedded by Sergei
    X_train_cache_path = Path(PREPROC_DATA_DIR).joinpath(f'X_train_embed_S.npy')

    # y_train are stored locally in a npy file labelled with NUM_OF_LABELS
    y_train_cache_path = Path(PREPROC_DATA_DIR).joinpath(f'y_train_{NUM_OF_LABELS}.npy')

    # y_train are stored locally in a npy file labelled with NUM_OF_LABELS
    y_labels_cache_path = Path(PREPROC_DATA_DIR).joinpath(f'y_labels_{NUM_OF_LABELS}.txt')

    # X_train is loaded locally if X_train_cache_path exists, else < TO IMPLEMENT >
    if X_train_cache_path.is_file():
        X_train = get_data_with_cache(X_train_cache_path)
    else:
        pass

    # y_train and y_labels are loaded locally if y_train_cache_path and y_labels_cache_path exist
    # else the preproc for the target is called
    if y_train_cache_path.is_file() and y_labels_cache_path.is_file:
        y_train = get_data_with_cache(y_train_cache_path)
        y_labels = get_data_with_cache(y_labels_cache_path)
    else:
        y_train, y_labels = encoding_target(train_terms,series_train_protein_ids,NUM_OF_LABELS)
        np.save(y_train_cache_path,y_train)
        np.save(y_labels_cache_path,y_labels)
