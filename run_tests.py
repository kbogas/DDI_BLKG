#!/usr/bin/python
# encoding:utf-8

"""
This is a sample script to generate the results as discussed in the paper.
First of all the required modules should be installed and the related embeddings downloaded in ./data/Embeddings folder.

The procedure followed merges one-by-one the embeddings for each method into one big dataframe,
while keeping track of the columns that correspond to each one. Finally, the 10-fold cv procedure with the inner 5-fold cv for hyperparameter tuning is followed.

Just run the script to invoke the main.
"""

import random
import copy
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from tqdm import tqdm_notebook
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split, RandomizedSearchCV
from tabulate import tabulate

# For deterministic purposes
random_state = 42
np.random.seed(random_state)
random.seed(random_state)

def load_data(model_names, path_to_embeddings_folder='./data/Embeddings/'):
    """
    Helper function to load the dataset in one big dataframe.
    Input:
        - model_names: list/iterable,
          iterable of strings containing the model names. Starting from DDI-BLKG for initialization
        - path_to_embeddings_folder: string,
          path to the downloaded embedding folder
    Output:
        - df_all: pd.DataFrame,
          one big dataframe containing one drug-pair per row. Its columns are the concatenated
          embeddings and other features of each drug pair (e.g. the name of the drugs in the pair,
          if it's AD-/LC-related, its truth_label etc.)
        - columns: dictionary,
          contains the column boundaries in the df_all variable, for each competing methodology
    """
    df_all, columns = load_ddi_blkg_and_dataset(path_to_embeddings_folder)
    for name_of_method in model_names[1:]:
        df_all, columns = append_features_for_drugs(df_all, columns, path_to_embeddings_folder, name_of_method)
    return df_all, columns

def load_ddi_blkg_and_dataset(path_to_embeddings_folder):
    """
    Helper function to load the drug pairs and the DDI-BLKG data.
    Input:
        - path_to_embeddings_folder: string,
          path to the downloaded embedding folder
    Output:
        - df_all: pd.DataFrame,
          one big dataframe containing one drug-pair per row. Its columns are the concatenated
          embeddings and other features of each drug pair (e.g. the name of the drugs in the pair,
          if it's AD-/LC-related, its truth_label etc.)
        - columns: dictionary,
          contains the column boundaries in the df_all variable, for each competing methodology
    """
    path_to_load = os.path.join(path_to_embeddings_folder, 'Dataset_and_BLKG.csv')
    df_all = pd.read_csv(path_to_load)
    df_all = df_all.drop('Path_Count', axis=1)
    df_all = df_all[df_all['Literature'] == 1]
    df_all = df_all.drop(['INTERACTS', 'Literature'], axis=1)
    columns = {'DDI-BLKG': [1, 105 + 1]}
    print(f'Loaded data and DDI-BLKG embeddings. Size {df_all.shape[0], df_all.shape[1]-5}')
    return df_all, columns

def append_features_for_drugs(df_all, columns, path_to_embeddings_folder, name_of_method):
    """
    Helper function to update df_all and columns with new features. This appends the new features in the existing dataframe and updates the columns dict as needed.
    Input:
        - df_all: pd.DataFrame,
          one big dataframe containing one drug-pair per row. Its columns are the concatenated
          embeddings and other features of each drug pair (e.g. the name of the drugs in the pair,
          if it's AD-/LC-related, its truth_label etc.)
        - columns: dictionary,
          contains the column boundaries in the df_all variable, for each competing methodology
        - path_to_embeddings_folder: string,
          path to the downloaded embedding folder
        - name_of_method: string,
          name of the methodoly to load its corresponding embedding file
    Output:
        - df_all: pd.DataFrame,
          one big dataframe containing one drug-pair per row. Its columns are the concatenated
          embeddings and other features of each drug pair (e.g. the name of the drugs in the pair,
          if it's AD-/LC-related, its truth_label etc.)
        - columns: dictionary,
          contains the column boundaries in the df_all variable, for each competing methodology
    """
    path_to_load = os.path.join(path_to_embeddings_folder, name_of_method)
    path_to_load += '.csv'
    df_cur_embeddings = pd.read_csv(path_to_load)
    df_cur_embeddings = df_cur_embeddings.iloc[df_all.index]
    df_all = df_all.merge(df_cur_embeddings, on='pair')
    if name_of_method == 'TransE':
        df_all.reset_index(drop=True, inplace=True)
        columns.update({'TransE_Emb': [110, 110 + 300]})
        columns.update({'TransE': [columns['TransE_Emb'][1], columns['TransE_Emb'][1] + 1]})
    elif name_of_method == 'HolE':
        columns.update({'HolE_Emb': [columns['TransE'][1], columns['TransE'][1] + 300]})
        columns.update({'HolE': [columns['HolE_Emb'][1], columns['HolE_Emb'][1] + 1]})
    elif name_of_method == 'DistMult':
        columns.update({'DistMult_Emb': [columns['HolE'][1], columns['HolE'][1] + 300]})
        columns.update({'DistMult': [columns['DistMult_Emb'][1], columns['DistMult_Emb'][1] + 1]})
    elif name_of_method == 'RESCAL':
        columns.update({'RESCAL_Emb': [columns['DistMult'][1], columns['DistMult'][1] + 200]})
        columns.update({'RESCAL': [columns['RESCAL_Emb'][1], columns['RESCAL_Emb'][1] + 1]})
    else:
        print('Method not implemented!You need to format the correspoding loading method!')
        raise NotImplementedError
    print('Appended %s embeddings. Size: (%d, %d)' % (name_of_method, df_cur_embeddings.shape[0], df_cur_embeddings.shape[1]-2))
    return df_all, columns

def generate_correct_data(df_all, related, balanced, truth_label):
    """
    Simple function to fetch the needed data from the big dataframe. This allows for fetching either AD,LC or all data. Also, it allows for down-sampling the negative class to have a
    balanced sample.
    Input:
        - df_all: pd.DataFrame,
          Dataframe containing the drug-pairs with their corresponding features and a column ("AD") denoting whether this is an AD or LC pair.
        - related: string or None,
          if related == 'AD' only ad-related drugs are kept, while if related == 'LC' only lc drugs
          are kept. If anything else, all drugs are kept.
        - balance: boolean,
          if True downsample the negative class to match the size of the positive class.
        - truth_label: string,
          the name of the column containing the truth labels about the interactivity of the pair
    Output:
        - df_data: pd.DataFrame,
        the fetched dataframe according to related and balance
        - details: dictionary,
        dictionary containing the details of the selection (i.e. if only 'AD' drugs where kept, whether they were balanced etc.)
    """
    df_cur = df_all.copy()
    details = {}
    if related == 'AD':
        df_cur = df_cur[df_cur['AD']==1]
        details['Related'] = "AD"
    elif related == 'LC':
        df_cur = df_cur[df_cur['LC']==1]
        details['Related'] = "LC"
    else:
        details['Related'] = "All"
    if balanced:
        details['Balanced'] = "Yes"
        numb_small = df_cur.groupby(truth_label).size().min()
        label_small = df_cur.groupby(truth_label).size().argmin()
        # print("The small class is: %d with :%d items" % (label_small, numb_small))
        df_pos = df_cur[df_cur[truth_label] == 1].sample(n=numb_small)
        df_neg = df_cur[df_cur[truth_label] == 0].sample(n=numb_small)
        df_data = pd.concat([df_pos, df_neg])
    else:
        details['Balanced'] = "No"
        df_data = df_cur.copy()
    return df_data, details


def get_scores(y, res2, res2_probas):
    """
    Scores for the evaluations of each method.
    Input:
        - y: iterable of length (N_samples),
          contains the truth labels of the samples
        - res2: iterable,
          contains the predicted labels of the samples. It is expected to be of length N_samples.
        - res2_probas: iterable,
          contains the predicted probabilities of the classes. It is expected to be of size
          N_samples x 2, where the [:, 1] column contains the probabilities of the positive class.
    Output:
        - dictionary, containing the measures needed.
    """

    if len(np.shape(res2_probas)) == 1:
        res2_probas = np.hstack((1 - res2_probas.reshape(-1, 1), res2_probas.reshape(-1, 1)))
    p, r, f, s = metrics.precision_recall_fscore_support(y, res2, pos_label=1)
    return {
        "Accuracy": metrics.accuracy_score(y, res2),
        "AUC": metrics.roc_auc_score(y, res2_probas[:, 1]),
        "AP": metrics.average_precision_score(y, res2_probas[:, 1]),
        "Precision": p[1],
        "Recall": r[1],
        "F1": f[1],
        "Positive Support": s[1],
        "Support": np.sum(s)
    }




def main():
    """
    Wrapper function to execute experiments.
    Currently all functionality inside this main.
    """
    # Name of the model names
    model_names = [
                   'DDI-BLKG',
                   'TransE',
                   'HolE',
                   'DistMult',
                   'RESCAL',
    ]
    # Load the data and features
    print('~'*50)
    print(f'Initializing data loading..')
    df_all, columns = load_data(model_names, './data/Embeddings')
    # For the nested cv procedures
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    use_inner_cv = True
    # Random Grid for param-tuTning
    n_estimators = [100, 300, 500, 1000]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(5, 110, num=5)]
    max_depth.append(None)
    min_samples_split = [2, 5, 8, 10, 15, 20]
    min_samples_leaf = [1, 2, 3, 4, 5, 10, 20]
    bootstrap = [True, False]
    criterion = ["gini", "entropy"]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   "criterion": criterion,
                   'random_state':[random_state]}
    # Label of the drug pair according to which drugbank
    truth_label = 'GT_503'
    combs = [
      (False, 1, 'AD'),
      (False, 1, 'LC'),
    ]
    results = []
    print('~'*50)
    print(f'Initializing 10-fold procedure..')
    for model_name in model_names:
        print(f'Begin 10-fold on model: {model_name}')
        for comb in combs:
            balanced, literature, related = comb
            df_data, details = generate_correct_data(df_all, related, balanced, truth_label)
            details['Model'] = model_name
            fold_count = 0
            print(f'\nData distribution for {related}: \n{df_data[truth_label].value_counts().to_string()}\n')
            for train_index, test_index in outer_cv.split(df_data, df_data[truth_label]):
                # rint(f'Fold: {fold_count}')
                fold_count += 1
                details_run = details.copy()
                details_run['Fold'] = fold_count
                cur_y = df_data[truth_label].values
                train_y = cur_y[train_index]
                test_y = cur_y[test_index]
                cur_X = df_data[df_data.columns[columns[model_name][0]:columns[model_name][1]]].values
                train_X = cur_X[train_index]
                test_X = cur_X[test_index]
                if use_inner_cv:
                    dt = RandomForestClassifier(n_jobs=1, random_state=random_state)
                    #gs = GridSearchCV(dt, param_grid=param_grid, cv=inner_cv, refit=True, n_jobs=-1, verbose=0)
                    gs = RandomizedSearchCV(estimator=dt, param_distributions=random_grid, n_iter=30, n_jobs=-1, cv=inner_cv, refit=True, random_state=random_state, verbose=0)
                    gs.fit(train_X, train_y)
                    dt = gs.best_estimator_
                else:
                    dt = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=random_state)
                    from sklearn.tree import DecisionTreeClassifier
                    dt = DecisionTreeClassifier()
                    dt.fit(train_X, train_y)
                pred_probas = dt.predict_proba(test_X)
                pred_classes = np.argmax(pred_probas, axis=1)
                res = get_scores(test_y, pred_classes, pred_probas)
                details_run.update(res)
                # print(res)
                # print(metrics.confusion_matrix(test_y, pred_classes))
                results.append(details_run)
        print('~' * 50)


    df = pd.DataFrame(results)[["F1", "AUC", "AP", 'Model']]
    df = df.groupby('Model').mean().sort_values("AUC", ascending=False)
    pd.options.display.float_format = '{:,.3f}'.format
    print(f'Results on the 10-fold procedure (mean scores reported):')
    print(tabulate(df, headers='keys', tablefmt='psql'))
    exit()

if __name__ == '__main__':
    main()
