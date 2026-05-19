#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:28:04 2025

@author: cormerod
"""

from model.cat2vec import DataFrameEmbeddingModel
from data.get_data import (get_embedding_genes,
                           get_full_go_terms,
                           get_pathways,
                           get_complexes)

config = {"full_go":{"function":get_full_go_terms,
                     "threshold":0,
                     "dimension":1024,
                     "hidden":2048,
                     "lr":1e-3,
                     "epochs":20,
                     "path":"data/generated/full_go_embedding.feather"},
          "complexes":{"function":get_complexes,
                       "threshold":0,
                       "dimension":1024,
                       "hidden":2048,
                       "lr":1e-3,
                       "epochs":20,
                       "path":"data/generated/complex_embedding.feather"},
         "pathways":{"function":get_pathways,
                     "threshold":0,
                     "dimension":1024,
                     "hidden":2048,
                     "lr":1e-3,
                     "epochs":20,
                     "path":"data/generated/pathway_embedding.feather"},
         }

def make_embeddings():
    
    for k,v in config.items():
        data = v['function'](threshold=v['threshold'])
        model = DataFrameEmbeddingModel(data, 
                                        dimension = v['dimension'], 
                                        hidden = v['hidden'])
        model.fit(lr=v['lr'], 
                  epochs=v['epochs'], 
                  early_stopping_patience=4)
        embedding = model.to_dataframe()
        embedding.to_feather(v['path'])
        