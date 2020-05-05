#!/usr/bin/python
# encoding:utf-8


import pandas as pd
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split


random_state = 42
np.random.seed(random_state)
random.seed(random_state)

cv_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)


### FOR RANDOM SEARCH
# Number of trees in random forest
n_estimators = [100]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
                'random_state':rs}



def generate_correct_data(df_all, related, balance, truth_label):
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
        print("The small class is: %d with :%d items" % (label_small, numb_small))
        df_pos = df_cur[df_cur[truth_label] == 1].sample(n=numb_small)
        df_neg = df_cur[df_cur[truth_label] == 0].sample(n=numb_small)
        df_data = pd.concat([df_pos, df_neg])
    else:
        details['Balanced'] = "No"
        df_data = df_cur.copy()
    return df_data, details


def get_scores(y, res2, res2_probas):
    """
    Precision, Recall, F1 for positive class.
    """
    #print(y.shape)
    #print(res2.shape)
    if len(np.shape(res2_probas)) == 1:
        res2_probas = np.hstack((1 - res2_probas.reshape(-1,1), res2_probas.reshape(-1,1)))
    p, r, f, s = metrics.precision_recall_fscore_support(y, res2, pos_label=1)
    return {
        "Accuracy":metrics.accuracy_score(y,res2),
        "AUC":metrics.roc_auc_score(y, res2_probas[:,1]),
        "AP":metrics.average_precision_score(y, res2_probas[:,1]),
        "Precision": p[1],
        "Recall": r[1],
        "F1": f[1],
        "Positive Support": s[1],
        "Support": np.sum(s)
    }



# Load drug-pairs and DDI_BLKG embeddings
df_all = pd.read_csv('./data/Dataset_and_BLKG.csv')
df_all.head()
df_all = df_all.drop('Path_Count', axis=1)
df_all = df_all[df_all['Literature']==1]
df_all = df_all.drop(['INTERACTS', 'Literature'], axis=1)


# NCSR columns
columns = {'NCSR':[1, 105 + 1]}
df_all[df_all.columns.values[columns['NCSR'][0]:columns['NCSR'][1]]]



# TRANSE

df_transe_embeddings = pd.read_csv("./data/TransE.csv")
print(df_transe_embeddings.shape)
df_transe_embeddings = df_transe_embeddings.iloc[df_all.index]
print(df_transe_embeddings.shape)



df_all = df_all.merge(df_transe_embeddings, on='pair')
columns.update({'TransE_Emb':[110, 110+300]})
columns.update({'TransE':[columns['TransE_Emb'][1],columns['TransE_Emb'][1]+1]})
print(df_all[df_all.columns.values[columns['TransE_Emb'][0]:columns['TransE_Emb'][1]]].head())
print(df_all[df_all.columns.values[columns['TransE'][0]:columns['TransE'][1]]].head())
del df_transe_embeddings
df_all.reset_index(drop=True, inplace=True)



# Hole
df_hole_embeddings = pd.read_csv("./Embedding_Methods/Literature_Embeddings/HolE_3_3.csv")
df_hole_embeddings = df_hole_embeddings.iloc[df_all.index]
print(df_hole_embeddings.shape)
df_all = df_all.merge(df_hole_embeddings, on='pair')
columns.update({'HolE_Emb':[columns['TransE'][1],columns['TransE'][1]+300]})
columns.update({'HolE':[columns['HolE_Emb'][1],columns['HolE_Emb'][1]+1]})
print(df_all[df_all.columns.values[columns['HolE_Emb'][0]:columns['HolE_Emb'][1]]])
print(df_all[df_all.columns.values[columns['HolE'][0]:columns['HolE'][1]]])
del df_hole_embeddings


# DistMult
df_dist_embeddings = pd.read_csv("./Embedding_Methods/Literature_Embeddings/DistMult_3_3.csv")
df_dist_embeddings = df_dist_embeddings.iloc[df_all.index]
print(df_dist_embeddings.shape)
df_all = df_all.merge(df_dist_embeddings, on='pair')
columns.update({'DistMult_Emb':[columns['HolE'][1],columns['HolE'][1]+300]})
columns.update({'DistMult':[columns['DistMult_Emb'][1],columns['DistMult_Emb'][1]+1]})
print(df_all[df_all.columns.values[columns['DistMult_Emb'][0]:columns['DistMult_Emb'][1]]])
print(df_all[df_all.columns.values[columns['DistMult'][0]:columns['DistMult'][1]]])
del df_dist_embeddings


# Rescal

df_rescal_embeddings = pd.read_csv("./Embedding_Methods/Literature_Embeddings/RESCAL_8_3.csv")
df_rescal_embeddings = df_rescal_embeddings.iloc[df_all.index]
print(df_rescal_embeddings.shape)
df_all = df_all.merge(df_rescal_embeddings, on='pair')
columns.update({'RESCAL_Emb':[columns['DistMult'][1],columns['DistMult'][1]+200]})
columns.update({'RESCAL':[columns['RESCAL_Emb'][1],columns['RESCAL_Emb'][1]+1]})
print(df_all[df_all.columns.values[columns['RESCAL_Emb'][0]:columns['RESCAL_Emb'][1]]])
print(df_all[df_all.columns.values[columns['RESCAL'][0]:columns['RESCAL'][1]]])
del df_rescal_embeddings


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
cv_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
truth_label = 'GT_531'
use_inner_cv = True
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
model_names = [
               'NCSR',
               'TransE_Emb',
               'HolE_Emb',
               'DistMult_Emb',
               'RESCAL_Emb',

#                'TransE',
#                'HolE',
#                'RESCAL',
#                'DistMult'
              ]
combs = [
#    (False, 1, 'All'),
  (False, 1, 'AD'),
  (False, 1, 'LC'),
# (True, 1, 'AD'),
# (True, 1, 'LC'),
#     (True, 1, 'All'),
#     (True, 1, 'AD'),
#     (True, 1, 'LC'),
#     (True, 1, 'All'),
#    (True, 0, 'All'),
#    (True, 0, 'LC'),
#    (True, 0, 'AD'),
]
results = []
for model_name in model_names:
    print(f'Model: {model_name}')
    for comb in combs:
        print(f'Comb: {comb}')
        balanced, literature, related = comb
        df_data, details = generate_correct_data(df_test, related, balanced, truth_label)
        #print(comb)
        details['Model'] = model_name
        fold_count = 0
        print(df_data[truth_label].value_counts())
        for train_index, test_index in cv_folds.split(df_data, df_data[truth_label]):
            print(f'Fold: {fold_count}')
            fold_count += 1
            details_run = details.copy()
            details_run['Fold'] = fold_count
            cur_y = df_data[truth_label].values
            train_y = cur_y[train_index]
            test_y = cur_y[test_index]
            if model_name == 'AMFP':
                pair_names = df_data.pair.values
                train_X = get_emb_from_pairs(pair_names[train_index], d_emb)
                test_X = get_emb_from_pairs(pair_names[test_index], d_emb)
            else:
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
                dt.fit(train_X, train_y)
            pred_probas = dt.predict_proba(test_X)
            pred_classes = np.argmax(pred_probas, axis=1)
            res = get_scores(test_y, pred_classes, pred_probas)
            details_run.update(res)
            print(res)
            print(metrics.confusion_matrix(test_y, pred_classes))
            results.append(details_run)
            print('~'*50)

