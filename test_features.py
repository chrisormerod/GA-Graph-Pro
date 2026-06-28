#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 07:25:12 2025

@author: cormerod
"""

import warnings 
import pandas as pd
import numpy as np
from data.get_data import (get_target_data,
                           get_embedding_genes, 
                           get_full_go_terms,
                           get_pathways,
                           get_complexes,
                           get_subcellular_location,
                           get_mouse_expression,
                           get_atlas_expression,
                           get_normal_atlas_expression,
                           get_tabula_muris_expression,
                           get_tabula_muris_sapiens_expression,
                           get_tabula_muris_senis_expression,
                           get_coprecipitation_adjacency,
                           get_metabolic_adjacency,
                           get_transcription_adjacency,
                           get_dephosphorylation_adjacency,
                           get_phosphorylation_adjacency)
from utils.data_utils import (set_index, 
                              merge_list,
                              create_kfold_splits,
                              restrict_dataframe,
                              select_features,
                              get_geometric_data_split)
from utils.metrics import get_row
from tqdm import tqdm
from model.nnlinear import NNLinear
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr
from collections import Counter
from graph.network_data import get_geometric_data

warnings.filterwarnings("ignore")


target_data = get_target_data(threshold=100)

genes=list(target_data.index)

feature = restrict_dataframe(merge_list([get_full_go_terms(genes, embedding= 0),
                                          get_complexes(genes, embedding=0),
                                          get_pathways(genes, embedding=0),
                                          get_subcellular_location(genes),
                                          get_mouse_expression(genes),
                                          get_normal_atlas_expression(genes),
                                          get_tabula_muris_expression(genes),
                                          get_tabula_muris_sapiens_expression(genes),
                                          get_tabula_muris_senis_expression(genes),
                                          get_coprecipitation_adjacency(genes),
                                          get_metabolic_adjacency(genes),
                                          get_transcription_adjacency(genes),
                                          get_phosphorylation_adjacency(genes),
                                          get_dephosphorylation_adjacency(genes)]),0)


data = get_geometric_data(target_data, genes=genes, split_number=0, selection_threshhold=2)
splits = create_kfold_splits(genes, n_splits = 10)

rows = []
feature_selection = select_features(split_number=0, threshold=2)
split = splits[0]

# target_data = get_target_data(threshold = 10)
genes = list(target_data.index)

feature = restrict_dataframe(merge_list([get_full_go_terms(genes, embedding= 0),
                                         get_complexes(genes, embedding=0),
                                         get_pathways(genes, embedding=0),
                                         get_subcellular_location(genes),
                                         get_mouse_expression(genes),
                                         get_normal_atlas_expression(genes),
                                         get_tabula_muris_expression(genes),
                                         get_tabula_muris_sapiens_expression(genes),
                                         get_tabula_muris_senis_expression(genes),
                                         get_coprecipitation_adjacency(genes),
                                         get_metabolic_adjacency(genes),
                                         get_transcription_adjacency(genes),
                                         get_phosphorylation_adjacency(genes),
                                         get_dephosphorylation_adjacency(genes)]),0)
embedding = get_embedding_genes(genes)
embedding.columns = [str(x) for x in embedding.columns]

splits = create_kfold_splits(genes)

results = {}

split_num = 0
split = splits[0]

feature_selection = select_features(split_num, threshold=2)
results[split_num] = []
for s in tqdm(target_data.columns):
    y_train = target_data.loc[split['train']][s]
    y_dev = target_data.loc[split['dev']][s]
    y_test = target_data.loc[split['test']][s]
    
    classifiers = {}
    for feature_type, features in feature_selection.items():
        X_train = feature.loc[split['train']][features]
        classifiers[feature_type] = LogisticRegression(class_weight="balanced")
        classifiers[feature_type].fit(X_train,y_train)
        break
    X_train = embedding.loc[split['train']]
    classifiers['embedding'] = LogisticRegression(class_weight="balanced")
    classifiers['embedding'].fit(X_train, y_train)
    
    probas = {}
    preds = {}
    for feature_type, features in feature_selection.items():
        X_test = feature.loc[split['test']][features]
        probas[feature_type] = classifiers[feature_type].predict_proba(X_test)
        preds[feature_type] = classifiers[feature_type].predict(X_test)
        break
    X_test =  embedding.loc[split['test']]
    probas['embedding'] = classifiers['embedding'].predict_proba(X_test)
    preds['embedding'] = classifiers['embedding'].predict(X_test)
    
    results_row = {'symptom':s}
    for feature_type,y_pred_test in preds.items():
        results_row.update(get_row(s,y_test, y_pred_test, set_name = f"test_{feature_type}"))
    y_ensemble_test = sum(list(probas.values())).argmax(-1)
    results_row.update(get_row(s,y_test, y_ensemble_test, set_name = f"test_ensemble"))
    results[split_num].append(results_row)
# pd.DataFrame(results[split_num]).to_csv(f"results/feature_selection_summary_{split_num}.csv")



train, dev, test = get_geometric_data_split(data, splits[0], target_data.columns[0])

X_train = train['genes:protein'].x.numpy()
X_dev = dev['genes:protein'].x.numpy()[len(X_train):,:]
X_test = test['genes:protein'].x.numpy()[len(train['genes:protein'].x):,:]


qbar = tqdm(zip(train['genes'].x.transpose(1,0), test['genes'].x.transpose(1,0), target_data.columns))

for y_train,y_test, s in qbar:
    y_train = y_train.numpy()
    y_test = y_test[len(train['genes'].x):]
    clf = LogisticRegression(class_weight="balanced")
    # y_train = target_data[s].loc[splits[0]['train']]
    
    # X_dev = dev['genes:protein'].x.numpy()[len(y_train):,:]
    # y_dev = target_data[s].loc[splits[0]['dev']]
    
    # X_test = test['genes:protein'].x.numpy()[len(y_train):,:]
    # y_test = 
    if sum(y_test) > 0 and sum(y_train):
        clf.fit(X_train, y_train)
        row = get_row(s, y_train, clf.predict(X_train), set_name = "train")
        row.update(get_row(s, y_test, clf.predict(X_test), set_name = "test"))
        rows.append(row)
    df = pd.DataFrame(rows)
    try:
        kp = sum(df['kappa_test']>=0.2)
    except:
        kp = 0
    qbar.desc =str( kp)