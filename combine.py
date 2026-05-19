# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 09:20:57 2026

@author: chris
"""
import json
import pandas as pd
from data import get_target_data
from graph.network_data import get_geometric_data
from utils.data_utils import create_kfold_splits
from utils.metrics import (compute_mcc_matrix,
                           get_row)
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from model.pava import parse_hpo_file, apply_pava

run_name = "final"

target_data = get_target_data(threshold = 10)
target_data = target_data.astype(int)

genes = list(target_data.index)
splits = create_kfold_splits(genes, dev_prop = 0.1)

split = 0
hidden = 512

train_targets = target_data.loc[splits[split]['train']]
dev_targets = target_data.loc[splits[split]['dev']]
test_targets = target_data.loc[splits[split]['test']]

data = get_geometric_data(target_data, genes=genes, split_number=0, selection_threshhold=10, hidden = hidden)

def get_data(run_name, split, symptom):
    dev = pd.read_csv(f"results/{run_name}/dev/dev_output_{split}_{symptom}.csv")
    test = pd.read_csv(f"results/{run_name}/test/test_output_{split}_{symptom}.csv")
    return dev, test

with open(f"results/{run_name}/log-{split}.json","r") as fp:
    results = json.load(fp)
    
ensemble_df = pd.DataFrame(columns = list(results))

for symptom in tqdm(list(results),total=len(results)):
    dev, test = get_data(run_name, split, symptom)    
    if len(dev) > 0:
        clf = LogisticRegression(class_weight="balanced")
        clf.fit(dev.values[:,1:], dev_targets[symptom].values)
        res = pd.DataFrame(clf.predict_proba(test.values[:,1:]), index = test_targets.index)
    else:
        res = pd.DataFrame(test[[f"nn:test:{symptom}_0",f"nn:test:{symptom}_1"]].values/2 
                           + test[[f"lr:test:{symptom}_0",f"lr:test:{symptom}_1"]].values/2, index = test_targets.index)
    ensemble_df[symptom] = res[1]
mcc_matrix = compute_mcc_matrix(train_targets)

all_nodes, roots = parse_hpo_file("data/raw/hp.obo")

thresh = 0.3

linked_df = pd.DataFrame(columns = list(results), index = test_targets.index)

for s1 in tqdm(list(results),total=len(results)):
    res = sum([mcc_matrix.loc[s1,s2]*ensemble_df[s2] for s2 in list(results) if mcc_matrix.loc[s1,s2]>=thresh])/sum([mcc_matrix.loc[s1,s2] for s2 in list(results) if mcc_matrix.loc[s1,s2]>=thresh])
    linked_df[s1] = res

ens = pd.DataFrame([get_row(s,test_targets[s], ensemble_df[s]>0.5) for s in ensemble_df.columns])
linked = pd.DataFrame([get_row(s,test_targets[s], linked_df[s]>0.5) for s in linked_df.columns])

pava_df = apply_pava(linked_df, all_nodes)

res_pava = pd.DataFrame([get_row(s, test_targets[s], pava_df[s]>0.5, set_name="pava") for s in pava_df.columns])
res_pava.to_csv(f"results/new_pava_0.csv")

# res_linked = pd.DataFrame([get_row(s, test_targets[s], linked_df[s]>0.5, set_name="linked") for s in linked_df.columns])
# res_linked.to_csv(f"results/linked_0.csv")

import pdb
pdb.set_trace()