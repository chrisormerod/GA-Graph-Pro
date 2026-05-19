#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 15:53:12 2025

@author: cormerod
"""
import os
os.chdir("..")
from data.get_data import (get_coprecipitation,
                           get_metabolic,
                           get_transcription,
                           get_phosphorylation,
                           get_dephosphorylation,
                           get_target_data)
from graph.randomwalk import MultiLayerRandomWalk
from sklearn.linear_model import LogisticRegression
import pandas as pd
from utils.data_utils import (create_kfold_splits, 
                              get_geometric_data_split)
from utils.data_utils import (set_index,
                              restrict_dataframe,
                              create_union_dataframe)
from utils.metrics import get_row
import numpy as np

target_data = get_target_data(threshold = 100)
target_data = target_data.astype(int)

genes = list(target_data.index)
splits = create_kfold_splits(genes, dev_prop = 0.0)


coprecipitation = get_coprecipitation()
transcription = get_transcription()
metabolic = get_metabolic()
phosphorylation = get_phosphorylation()
dephosphorylation = get_dephosphorylation()

coprecipitation['value']=1
transcription['value']=1
metabolic['value']=1
phosphorylation['value']=1
dephosphorylation['value']=1

coprecipitation_adj_df = (coprecipitation.pivot_table(index = "gene_a", columns = "gene_b", values = "value").fillna(0) + coprecipitation.pivot_table(index = "gene_b", columns = "gene_a", values = "value").fillna(False)).fillna(False).astype(bool)
transcription_adj_df = create_union_dataframe(transcription.pivot_table(index = "gene_a", columns = "gene_b", values = "value")).fillna(False).astype(bool)
metabolic_adj_df = create_union_dataframe(metabolic.pivot_table(index = "gene_a", columns = "gene_b", values = "value")).fillna(False).astype(bool)
phosphorylation_adj_df = create_union_dataframe(phosphorylation.pivot_table(index = "gene_a", columns = "gene_b", values = "value")).fillna(False).astype(bool)
dephosphorylation_adj_df = create_union_dataframe(dephosphorylation.pivot_table(index = "gene_a", columns = "gene_b", values = "value")).fillna(False).astype(bool)


mlrw = MultiLayerRandomWalk(
    embedding_dim=256,
    walk_length=5,
    num_walks=10,
    inter_layer_prob=0.1
)

super_index = sorted(list(set(list(coprecipitation_adj_df.index) + list(transcription_adj_df.index) + list(metabolic_adj_df.index) + list(phosphorylation_adj_df.index) + list(dephosphorylation_adj_df.index))))

M_coprecipitation = coprecipitation_adj_df.reindex(index = super_index, fill_value=False).reindex(columns = super_index, fill_value=False).values
M_transcription = transcription_adj_df.reindex(index = super_index, fill_value=False).reindex(columns = super_index, fill_value=False).values
M_metabolic = metabolic_adj_df.reindex(index = super_index, fill_value=False).reindex(columns = super_index, fill_value=False).values
M_phosphorylation = phosphorylation_adj_df.reindex(index = super_index, fill_value=False).reindex(columns = super_index, fill_value=False).values
M_dephosphorylation = dephosphorylation_adj_df.reindex(index = super_index, fill_value=False).reindex(columns = super_index, fill_value=False).values

mlrw.add_network("coprecipitation", M_coprecipitation)
mlrw.add_network("transcription", M_transcription)
mlrw.add_network("metabolic", M_metabolic)
mlrw.add_network("phosphorylation", M_phosphorylation)
mlrw.add_network("dephosphorylation", M_dephosphorylation)

results = mlrw.compute_embeddings(range(26234))

M = np.concatenate([results['layer_embeddings'][k] for k in mlrw.networks],1)

features = pd.DataFrame(M, index = super_index)
features.columns = sum([[f'{l}_{i}' for i in range(253)] for l in results['layer_embeddings'].keys()],list())


# X_train = features.loc[splits[0]['train']]
# X_test = features.loc[splits[0]['test']]

# rows = []

# for s in target_data.columns:
#     y_train = target_data.loc[splits[0]['train'],s]
#     y_test = target_data.loc[splits[0]['test'],s]
#     clf = LogisticRegression(class_weight="balanced")
    
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     metrics = get_row(s, y_test, y_pred, set_name="test")
#     rows.append(metrics)