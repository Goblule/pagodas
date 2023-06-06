import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import progressbar

from Bio import SeqIO


def encoding_target(train_terms: pd.DataFrame, # raw train terms from train_terms.tsv file
                    series_train_protein_ids: pd.Series, # series containing the unique proteins IDs
                    num_of_labels: int, # number of most frequent GO term IDs
                    ) -> pd.DataFrame : # encoded target

                    # Take value counts in descending order and fetch first 1500 `GO term ID` as labels
                    labels = train_terms['term'].value_counts().index[:num_of_labels].tolist()

                    # Fetch the train_terms data for the relevant labels only
                    train_terms_updated = train_terms.loc[train_terms['term'].isin(labels)]

                    train_size = len(series_train_protein_ids) # 142246
                    train_labels = np.zeros((train_size,num_of_labels))

                    # Setup progressbar settings
                    bar = progressbar.ProgressBar(maxval=num_of_labels, \
                                                  widgets=[progressbar.Bar('=', 'Encoding Target [', ']'), ' ', progressbar.Percentage()])

                    # Loop through each label
                    for i in range(num_of_labels):

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

                    # Convert train_Y numpy into pandas dataframe
                    labels_df = pd.DataFrame(data = train_labels, columns = labels)

                    return labels_df
