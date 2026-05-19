#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:14:09 2025

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
                              select_features)
from utils.metrics import get_row
from tqdm import tqdm
from model.nnlinear import NNLinear
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr
from collections import Counter

warnings.filterwarnings("ignore")

target_data = get_target_data(threshold = 10)
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

for split_num, split in enumerate(splits):
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
            
        X_train = embedding.loc[split['train']]
        classifiers['embedding'] = LogisticRegression(class_weight="balanced")
        classifiers['embedding'].fit(X_train, y_train)
        
        probas = {}
        preds = {}
        for feature_type, features in feature_selection.items():
            X_test = feature.loc[split['test']][features]
            probas[feature_type] = classifiers[feature_type].predict_proba(X_test)
            preds[feature_type] = classifiers[feature_type].predict(X_test)
        X_test =  embedding.loc[split['test']]
        probas['embedding'] = classifiers['embedding'].predict_proba(X_test)
        preds['embedding'] = classifiers['embedding'].predict(X_test)
        
        results_row = {'symptom':s}
        for feature_type,y_pred_test in preds.items():
            results_row.update(get_row(s,y_test, y_pred_test, set_name = f"test_{feature_type}"))
        y_ensemble_test = sum(list(probas.values())).argmax(-1)
        results_row.update(get_row(s,y_test, y_ensemble_test, set_name = f"test_ensemble"))
        results[split_num].append(results_row)
    pd.DataFrame(results[split_num]).to_csv(f"results/feature_selection_summary_{split_num}.csv")
