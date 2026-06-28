#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:14:09 2025

@author: cormerod
"""

import warnings 
import pandas as pd
import numpy as np
import json
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
                              restrict_dataframe)
from utils.metrics import get_row
from collections import Counter
from tqdm import tqdm
from model.nnlinear import NNLinear
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

target_data = get_target_data(threshold = 10)
target_data = target_data.astype(int)
# target_data = target_data.drop("HP:0000001", axis=1)

genes = list(target_data.index)

gene_feature = restrict_dataframe(merge_list([get_full_go_terms(genes, embedding= 0),
                                          get_complexes(genes, embedding=0),
                                          get_pathways(genes, embedding=0),
                                          get_subcellular_location(genes)]),1)

exp_feature = restrict_dataframe(merge_list([get_mouse_expression(genes),
                                          get_normal_atlas_expression(genes),
                                          get_tabula_muris_expression(genes),
                                          get_tabula_muris_sapiens_expression(genes),
                                          get_tabula_muris_senis_expression(genes)]),1)

net_feature = restrict_dataframe(merge_list([get_coprecipitation_adjacency(genes),
                                          get_metabolic_adjacency(genes),
                                          get_transcription_adjacency(genes),
                                          get_phosphorylation_adjacency(genes),
                                          get_dephosphorylation_adjacency(genes)]),1)

splits = create_kfold_splits(genes, dev_prop = 0.2)

def feature_selection(feature, filename, max_features=100):
    output = {}
    for split_number, split in enumerate(splits):
        X_train = feature.loc[split['train']]
        X_dev = feature.loc[split['dev']]
        X_test = feature.loc[split['test']]
        rows = []
        qbar = tqdm(target_data.columns)
        for s in qbar:
                y_train = target_data[s].loc[split['train']]
                y_dev = target_data[s].loc[split['dev']]
                y_test = target_data[s].loc[split['test']]
                
                #import pdb
                #pdb.set_trace()
                
                contributions = {c:cohen_kappa_score(y_train, X_train[c].astype(int)) for c in X_train.columns}
                order = sorted(contributions.items(), key = lambda x:-np.abs(x[1]))
                
                best_i = 1
                best_mcc = -1
    
                for i in range(1,max_features):
                    cols = [x[0] for x in order[:i]]
                    Z_train = X_train[cols]
                    Z_dev = X_dev[cols]
                    Z_test = X_test[cols]
                    
                    model = LogisticRegression(class_weight="balanced")

                    model.fit(Z_train,y_train)
                    y_pred_dev = model.predict(Z_dev)
                    met = get_row("",y_pred_dev,y_dev,set_name = "dev")
                    # import pdb
                    # pdb.set_trace()
                    if met['matthews_dev'] > best_mcc:
                        best_i = i
                        best_mcc = met['matthews_dev']
            
                cols = [x[0] for x in order[:best_i]]
                Z_train = X_train[cols]
                Z_dev = X_dev[cols]
                Z_test = X_test[cols]
            
                model = LogisticRegression(class_weight="balanced")
                model.fit(Z_train,y_train)
                
                y_proba_dev = model.predict_proba(Z_dev)
                row = get_row(s, y_dev, y_proba_dev.argmax(-1), set_name = "dev")
                row['features'] = cols
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(f"results/feature_selection_mcc_{filename}_{split_number}.csv")
        results = sum((Counter(x) for x in df[df['matthews_dev']>=0.2]['features']), Counter())  
        output[split_number] = results
        with open(f"{filename}.json", "w") as fp:
            json.dump(output, fp)
    return output

feature_selection(gene_feature, "gene")
feature_selection(exp_feature, "expression")
feature_selection(net_feature, "network")
