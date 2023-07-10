# 🏯 Pagodas
## Protein Annotation _by_ Gene Ontology _through_ Deep _learning_ Automated System

## Contributors
[Julien Tetar](https://github.com/Goblule),
[Erika Fallacara](https://github.com/erikafallacara),
[Victor M'Baye](https://github.com/VeMBe06)

<!-- TABLE OF CONTENTS -->
<!--  <details>  -->
<!--  <summary>Table of Contents</summary>  -->
## Table of Contents
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#environmental-variables">Environmental Variables</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
<!-- </details>  -->

<!-- ABOUT THE PROJECT -->
## About The Project

The project follows the [Kaggle competition](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction) organized by the Critical Assessment of protein Function Annotation (CAFA). The goal of this competition is to predict the functions of a set of proteins from their amino acids sequences. The functions of the proteins are associated with the Gene Ontology (GO) terms, which are divided in the following three subontologies:

   - Molecular Function (MF)
   - Biological Process  (BP)
   - Cellular Component (CC)

The features (**INPUT**) are text sequences corresponding to the amino acids sequences:

$$
\verb|X = amino acids sequence|
$$

The target (**OUTPUT**) are the probabilities associated to each GO term.

$$
\verb|y = GO(MF,BP,CC)|
$$

<!-- GETTING STARTED -->
## Getting Started

The package can be run locally or on Google Cloud Platform (GCP). If using this second option, the GCP project ID, the GCP region and the GCP bucket need to be specified in the enviromental variables `GCP_PROJECT`, `GCP_REGION` and `BUCKET_NAME`, respectively.

### Prerequisites

The required libraries can be installed via the command
```
pip install -r requirements.txt
```

### Installation

To install directly the package, you can run:

```
make reinstall_package
```

This will reinstall the requirements and the package.

### Environmental Variables

The environmental variables cab be updated using `direnv`. The environmental variables can be specified in a `.env` file as the following:
```
# PREPROC
NUM_OF_FEATS=
NUM_OF_LABELS=

# GCP PROJECT
GCP_PROJECT=
GCP_REGION=

# STORAGE DATA (local, gcs)
STORAGE_DATA_KEY=

# STORAGE MODEL (local, gcs)
STORAGE_MODEL_KEY=

# GCP CLOUD STORAGE
BUCKET_NAME=

# MODEL
MODEL_PROD_NAME=
```

where:
- `NUM_OF_FEATS` is the dimension of the $
\verb|X| $ vector embedded;
- `NUM_OF_LABELS` is the dimension of the $
\verb|y| $ vector encoded;
- `GCP_PROJECT` and `GCP_REGION` are the GCP project ID and region, respectively;
- `STORAGE_DATA_KEY` is a flag used to indicate if the data are stored locally (`STORAGE_DATA_KEY=local`) or on the cloud (`STORAGE_DATA_KEY=gcs`);
- `STORAGE_MODEL_KEY` is a flag used to indicate if the model is stored locally (`STORAGE_MODEL_KEY=local`) or on the cloud (`STORAGE_MODEL_KEY=gcs`);
- `BUCKET_NAME` is the name of the GCP bucket;
- `MODEL_PROD_NAME` is the name of the chosen model in production.

<!-- USAGE EXAMPLES -->
## Usage
The package options are available via:
```
make
```
With the following options appearing on your screen:
```
To install pagodas package, type:
make reinstall_package

Package options
  reset_local_files             clean the preproc_data storage
  run_preprocess                run preprocessing on raw data
  run_train_custom_model        train a model defined by the user
  run_predict                   run prediction on new data
```

To reset all the local files and clean your cache, type:

```
make reset_local_files
```

To perform the preprocessing on raw data, i.e. embedding the features and encoding the target, type:

```
make run_preprocess
```

To train a custom model (**option still to be implemented**), type:

```
make run_train_custom_model
```

To perform a prediction on new data, type:

```
make run_predict
```

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

We acknowledge [Sergei Fironov](https://www.kaggle.com/sergeifironov) for making available the
[T5 data embedded ](https://www.kaggle.com/code/sergeifironov/t5embeds-calculation-only-few-samples).
