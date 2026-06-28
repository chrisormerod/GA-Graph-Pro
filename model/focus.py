#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:14:09 2025

@author: cormerod
"""

import warnings 
import pandas as pd
from data.get_data import (get_target_data,
                           get_embedding_genes, 
                           get_full_go_terms,
                           get_pathways,
                           get_complexes)
from utils.data_utils import (set_index, 
                              merge_list,
                              create_kfold_splits,
                              restrict_dataframe)
from tqdm import tqdm
from model.nnlinear import NNLinear

target_data = get_target_data(threshold = 10)
genes = list(target_data.index)

feature = restrict_dataframe(merge_list([get_complexes(genes),
                                         get_pathways(genes),
                                         get_full_go_terms(genes)]),0)

splits = create_kfold_splits(genes)

split =  splits[0]
X_train = feature.loc[split['train']]
X_dev = feature.loc[split['dev']]
X_test = feature.loc[split['test']]


