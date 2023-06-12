import pandas as pd
import numpy as np
import os, re
import matplotlib.pyplot as plt
import seaborn as sns
import progressbar

from Bio import SeqIO
from params import *

import sentencepiece
from transformers import T5Tokenizer, TFT5EncoderModel

def encoding_target(train_terms: pd.DataFrame, # raw train terms from train_terms.tsv file
                    series_train_protein_ids: pd.Series, # series containing the unique proteins IDs
                    ) -> pd.DataFrame : # encoded target

                    # Take value counts in descending order and fetch first 1500 `GO term ID` as labels
                    labels = train_terms['term'].value_counts().index[:NUM_OF_LABELS].to_numpy()


                    # Fetch the train_terms data for the relevant labels only
                    train_terms_updated = train_terms.loc[train_terms['term'].isin(labels)]
                    train_size = len(series_train_protein_ids) # 142246
                    train_labels = np.zeros((train_size,NUM_OF_LABELS))

                    # Setup progressbar settings
                    bar = progressbar.ProgressBar(maxval=NUM_OF_LABELS, \
                                                  widgets=[progressbar.Bar('=', 'Encoding Target [', ']'), ' ', progressbar.Percentage()])

                    # Start the bar
                    bar.start()

                    # Loop through each label
                    for i in range(NUM_OF_LABELS):


                      # For each label, fetch the corresponding train_terms data
                      n_train_terms = train_terms_updated[train_terms_updated['term'] ==  labels[i]]

                      # Fetch all the unique EntryId aka proteins related to the current label(GO term ID)
                      label_related_proteins = n_train_terms['EntryID'].unique()

                      # In the series_train_protein_ids pandas series, if a protein is related
                      # to the current label, then mark it as 1, else 0.
                      # Replace the ith column of train_Y with with that pandas series.
                      train_labels[:,i] =  series_train_protein_ids.isin(label_related_proteins).astype(float)

                      # Progress bar percentage increase
                      bar.update(i+1)

                    # Notify the end of progress bar
                    bar.finish()

                    return train_labels, labels



def get_embedding( sequence : str,
                   embedding_model=None,
                   tokenizer=None) -> np.ndarray:
    """
    Function that generates the embeddings for a SINGLE protein sequence.
    Input = protein fasta sequence (str)
    Returns vector of size 1024
    The current version of this function uses TensorFlow with Pytorch weights (native
    function to the tranformers library); however, if need be, we can switch to
    regular pytorch
    """

    # If tokenizer not instantiated, init tokenizer
    if not tokenizer:
        print("Tokenizer not loaded yet!")
        print('\n Loading T5 tokenizer...')
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)


    # If model not instantiated, init model
    if not embedding_model:
        print("Embedding model not loaded yet!")
        print("Loading T5 encoding model...")
        embedding_model = TFT5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_bfd", from_pt=True)

    # Replace rare amino acids in sequence with X (aka "any")
    seq_processed = " ".join(re.sub(r'[UZOB]', 'X', sequence))

    # Encode sequence with tokenizer
    ids = tokenizer.batch_encode_plus(seq_processed,
                                      add_special_tokens=True,
                                      padding="longest",
                                      return_tensors='tf')

    # Separate ids into input_ids + attention_mask
    input_ids = ids['input_ids']
    attention_mask = ids['attention_mask']

    # Generate embeddings
    embedding_repr = embedding_model(input_ids=input_ids, attention_mask=attention_mask)


    # Extract residue embeddings for the first ([0,:]) sequence
    # in the batch and remove padded & special tokens ([0,:7])
    embedding_output = [np.mean(e, axis=0) for e in embedding_repr.last_hidden_state] # [0] is the embedding output, others are attention layers

    return embedding_output
