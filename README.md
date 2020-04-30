# Drug-Drug Interaction Prediction on a Biomedical Literature Knowledge Graph (DDI-BLKG)
Model drug-drug interactions (DDIs) based on a KG graph and use them to predict whether a DDI is probable.

In this repository exists  a sample script (*aime2020_results.py*) to replicate the results comparing the different methodologies is provided.

Moreover, a notebook for data exploration  is provided in the examples folder.

For the full workflow of creating the knowledge graph one should also visit
the [Biomedical Knowledge Integration repo](https://github.com/tasosnent/Biomedical-Knowledge-Integration) for setting up the Harvesters and the Extraction/Indexing Tools.

This is a work in progress. Contact the creator for anything not reported yet.

### Dataset:

In order to run the script and the notebook you will need to download the preprocessed datasets and place them in the *data* folder.

More specifically you need to download:

### Requirements:
Install the requirements

pip install -r requirements.txt

### Module:

You can also install the module if you would like to check it out from ipython.
```sh
git clone this_project
cd projectfolder
pip install --user .
```


## Example usage:
Run the following to generate the results as reported in our studies:
```python
python aime2020_results.py
```

## Tests

Currently no tests supported.

## Questions/Errors
Bougiatiotis Konstantinos, NCSR ‘DEMOKRITOS’ E-mail: bogas.ko@gmail.com

