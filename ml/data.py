
import numpy as np
import pandas as pd
import networkx
import obonet

from Bio import SeqIO
from pathlib import Path
from params import *
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
            # lengths = []
            for entry in SeqIO.parse(fastafile, 'fasta'):
                headers.append(entry.description)
                sequences.append(entry.seq)
                entries.append(entry)
                ids.append(entry.id)
            # Convert sequences (list of chars) to strings
            sequences = [str(x) for x in sequences]
            # Create dataframe
            df_fasta = pd.DataFrame({'ids':ids, 'headers':headers, 'seq':sequences})

    if STORAGE_DATA_KEY == 'gcs':
        # Path of train terms and fasta file
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
            # lengths = []
            for entry in SeqIO.parse(fastafile, 'fasta'):
                headers.append(entry.description)
                sequences.append(entry.seq)
                entries.append(entry)
                ids.append(entry.id)
            # Convert sequences (list of chars) to strings
            sequences = [str(x) for x in sequences]
            # Create dataframe
            df_fasta = pd.DataFrame({'ids':ids, 'headers':headers, 'seq':sequences})

    return df_fasta

def load_raw_train_terms() -> pd.DataFrame :

    '''
    This function loads the raw train terms either from the local directory or from the gcs cloud.
    The option is specified via the environment variable STORAGE_DATA_KEY (local, gcs).
    '''

    if STORAGE_DATA_KEY == 'local':
        train_terms_file = Path(RAW_DATA_DIR).joinpath('train_terms.tsv')
        train_terms = pd.read_csv(train_terms_file,sep='\t')

    if STORAGE_DATA_KEY == 'gcs':
        # Path of train terms and fasta file
        train_terms_file = 'raw_data/Train/train_terms.tsv'
        # Initialize client
        client = storage.Client()
        # Get bucket
        bucket = client.get_bucket(BUCKET_NAME)
        # Get blob
        blob_train_terms = bucket.get_blob(train_terms_file)
        # Define path
        train_terms_bucket_file = os.path.join(f'gs://{BUCKET_NAME}', blob_train_terms.name) # Define path
        # Read dataframe
        train_terms = pd.read_csv(train_terms_bucket_file,sep='\t')

    return train_terms


def get_data_with_cache(cache_path) -> np.array:
  print(f"\nLoading data from local npy file {cache_path} ...")
  array = np.load(cache_path,allow_pickle=True)
  print(f"✅ Data loaded, with shape {array.shape}")
  return array
