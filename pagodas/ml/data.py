
import numpy as np
import pandas as pd
import networkx
import obonet

from Bio import SeqIO
from pathlib import Path
from pagodas.params import *
from google.cloud import storage


def load_raw_obo_file() -> tuple :

    '''
    This function is implemented for the moment locally
    '''

    obo_file = Path(RAW_DATA_DIR).joinpath('go-basic.obo')

    # Read the gene ontology
    graph = obonet.read_obo(obo_file)

    # Number of nodes
    len(graph)

    # Number of edges
    graph.number_of_edges()

    # Check if the ontology is a DAG
    networkx.is_directed_acyclic_graph(graph)

    # Mapping from term ID to name
    id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}

    return graph, id_to_name

def load_raw_fasta_file() -> pd.DataFrame:

    '''
    This function reads the fasta file either from the local directory or from the gcs cloud.
    The option is specified via the environment variable STORAGE_DATA_KEY (local, gcs).
    '''

    if STORAGE_DATA_KEY == 'local':
        train_seq_file = Path(RAW_DATA_DIR).joinpath('train_sequences.fasta')
        with open(train_seq_file) as fastafile:
            headers = []
            sequences = []
            ids= []
            entries = []
            lengths = []

            for entry in SeqIO.parse(fastafile, 'fasta'):
                headers.append(entry.description)
                sequences.append(entry.seq)
                entries.append(entry)
                ids.append(entry.id)
            # Convert sequences (list of chars) to strings
            sequences = [str(x) for x in sequences]
            lengths = [len(x) for x in sequences]
            # Create dataframe
            df_fasta = pd.DataFrame({'id':id, 'header':headers, 'seq':sequences, 'length':lengths})

    if STORAGE_DATA_KEY == 'gcs':
        # Path of fasta file
        train_seq_file = 'raw_data/Train/train_sequences.fasta'
        # Initialize client
        client = storage.Client()
        # Get bucket
        bucket = client.get_bucket(BUCKET_NAME)
        # Get blob
        blob_train_seq = bucket.get_blob(train_seq_file)
        # Read blob
        with blob_train_seq.open('r') as fastafile:
            headers = []
            sequences = []
            ids= []
            entries = []
            lengths = []

            for entry in SeqIO.parse(fastafile, 'fasta'):
                headers.append(entry.description)
                sequences.append(entry.seq)
                entries.append(entry)
                ids.append(entry.id)
            # Convert sequences (list of chars) to strings
            sequences = [str(x) for x in sequences]
            # Create dataframe
            df_fasta = pd.DataFrame({'id':ids, 'header':headers, 'seq':sequences, 'length':lengths})

    return df_fasta

def clean_raw_fasta_df(df_fasta: pd.DataFrame) -> pd.DataFrame :
    # Clean the original fasta dataframe
    df = df_fasta.copy()
    df_dropped = df.drop_duplicates(subset=['seq'])
    return df_dropped

def load_raw_train_terms() -> pd.DataFrame :

    '''
    This function loads the raw train terms either from the local directory or from the gcs cloud.
    The option is specified via the environment variable STORAGE_DATA_KEY (local, gcs).
    '''

    if STORAGE_DATA_KEY == 'local':
        train_terms_file = Path(RAW_DATA_DIR).joinpath('train_terms.tsv')
        train_terms = pd.read_csv(train_terms_file,sep='\t')

    if STORAGE_DATA_KEY == 'gcs':
        # Path of train terms
        train_terms_file = 'raw_data/Train/train_terms.tsv'
        # Initialize client
        client = storage.Client()
        # Get bucket
        bucket = client.get_bucket(BUCKET_NAME)
        # Get blob
        blob_train_terms = bucket.get_blob(train_terms_file)
        # Define path
        train_terms_bucket_file = os.path.join(f'gs://{BUCKET_NAME}', blob_train_terms.name)
        # Read dataframe
        train_terms = pd.read_csv(train_terms_bucket_file,sep='\t')

    return train_terms

def get_preproc_data(data_filename: str) -> pd.DataFrame:

    '''
    This function loads arrays of preproc data either from the local directory or from the gcs cloud.
    The option is specified via the environment variable STORAGE_DATA_KEY (local, gcs).
    '''

    if STORAGE_DATA_KEY == 'local':
        cache_path = Path(PREPROC_DATA_DIR).joinpath(data_filename)
        print(f"\nLoading local csv file {data_filename} ...")
        data = pd.read_csv(cache_path,sep=',')

    if STORAGE_DATA_KEY == 'gcs':
        # Path of array file
        array_file = f'preproc_data/{data_filename}'
        print(f"\nLoading from gcs cloud csv file {data_filename} ...")
        # Initialize client
        client = storage.Client()
        # Get bucket
        bucket = client.get_bucket(BUCKET_NAME)
        # Get blob
        blob = bucket.get_blob(array_file)
        # Define path
        data_bucket_file = os.path.join(f'gs://{BUCKET_NAME}', blob.name)
        # Read dataframe
        data = pd.read_csv(data_bucket_file)

    return data

def save_preproc_data(array: np.array, array_filename: str) -> None :

    '''
    This function saves arrays of preproc data (format csv) in the local directory and in the gcs cloud
    (if specified via the environment variable STORAGE_DATA_KEY == gcs).
    '''

    #if STORAGE_DATA_KEY == 'local':
    cache_path = Path(PREPROC_DATA_DIR).joinpath(array_filename)
    pd.DataFrame(array).to_csv(cache_path,index=False)
    #np.save(cache_path,array)
    print(f"✅ {array_filename} saved locally")

    # Save array on gsc
    if STORAGE_DATA_KEY == 'gcs':
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f'preproc_data/{array_filename}')
        blob.upload_from_filename(cache_path)
        print(f"✅ {array_filename} saved to GCS")
