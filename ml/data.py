
import numpy as np
import pandas as pd
import obonet
import networkx

from Bio import SeqIO
from pathlib import Path

def read_fasta_file(train_seq_file: str) -> pd.DataFrame:

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

    return df_fasta

def read_obo_file(obo_file: str) -> tuple :

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


# load raw data
def load_raw_data_local(RAW_DATA_DIR: str) -> tuple :
    train_terms_file = Path(RAW_DATA_DIR).joinpath("train_terms.tsv")
    train_seq_file = Path(RAW_DATA_DIR).joinpath("train_sequences.fasta")
    train_terms = pd.read_csv(train_terms_file,sep='\t')
    train_seq = read_fasta_file(train_seq_file)
    return train_terms, train_seq

def get_data_with_cache(cache_path) -> np.array:
  print(f"\nLoading data from local npy file {cache_path} ...")
  array = np.load(cache_path,allow_pickle=True)
  print(f"✅ Data loaded, with shape {array.shape}")
  return array

