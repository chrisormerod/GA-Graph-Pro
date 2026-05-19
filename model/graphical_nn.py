#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 05:40:28 2025

@author: cormerod
"""
# import dgl
# from dgl.nn import HeteroLinear
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.data import HeteroData

class GeometricNeuralNetwork(nn.Module):
    
    def __init__(self, 
                 data : HeteroData,
                 hidden : int, 
                 dropout : float = 0.0):
        super().__init__()
        self.expressions = data.expressions
        self.gene_features = data['genes'].features
        self.hidden = hidden
        self.dropout = dropout
        # Gene Layers
        self.add_gene_layers(data)
        self.add_tissue_layers(data)
        
        
    def add_gene_layers(self, data):
        self.logistic_regression = nn.Linear(len(data['genes'].features), 1)
        self.gene_logistic_regression_layers = nn.ModuleDict()
        
        self.drop = nn.Dropout(p=self.dropout)
        
        for f in data['genes'].features:
            self.gene_logistic_regression_layers[f] = nn.Linear(in_features = (data[f'genes:{f}'].x).shape[1], 
                                                                out_features = 2)
        
        self.gene_layers = nn.ModuleDict()

        self.gene_network_layers = nn.ModuleDict()
        
        for f in data['genes'].features:
            self.gene_layers[f] = GATv2Conv(in_channels = (data[f'genes:{f}'].x).shape[1],
                                            out_channels = self.hidden)
            
        self.att_in = nn.Linear(in_features= self.hidden, out_features=self.hidden)   
        # self.att_in = nn.LayerNorm(normalized_shape=self.hidden)
        for edge_type in data['genes'].edges:
            self.gene_network_layers[edge_type] = GATv2Conv(in_channels = self.hidden, 
                                                            out_channels = self.hidden,
                                                            dropout = self.dropout)
        # self.att_out = nn.LayerNorm(normalized_shape=self.hidden)
        
        self.att_out = nn.Linear(in_features= self.hidden, out_features=self.hidden)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features = self.hidden, 
                      out_features = self.hidden),
            nn.ReLU(),
            nn.Dropout(p = self.dropout),
            nn.Linear(in_features = self.hidden, 
                      out_features = 2),
        )
                
    def add_tissue_layers(self, data):
        self.gene_tissue_layers = nn.ModuleDict()
        self.tissue_symptom_layers = nn.ModuleDict()

        for group in self.expressions:
            self.gene_tissue_layers[group] = HeteroConv({("genes","",f"tissues:{group}"): GATv2Conv(self.hidden, 
                                                                                                    self.hidden,
                                                                                                    dropout = self.dropout, 
                                                                                                    add_self_loops =False)}, 
                                                        aggr='sum')
            self.tissue_symptom_layers[group] = HeteroConv({(f"tissues:{group}","","symptoms"): GATv2Conv(self.hidden, 
                                                                                                          self.hidden, 
                                                                                                          dropout = self.dropout,
                                                                                                          add_self_loops =False)},
                                                           aggr='sum')
            
        
    def forward(self, data):
        device = self.logistic_regression.weight.device
        
        lr_outputs = []
        for f in data['genes'].features:
            X = self.drop(data[f'genes:{f}'].x)
            lr_outputs.append(self.gene_logistic_regression_layers[f](X))
        lr_X = sum(lr_outputs)/len(lr_outputs)
        
        gene_representation = data['genes'].x * torch.zeros((1,self.hidden), device=device)
        for f in data['genes'].features:
            gene_representation += self.gene_layers[f](data[f'genes:{f}'].x,
                                                        data[f'genes:{f}',"",'genes'].edge_index)
            # gene_representation += self.gene_layers[f](data[f'genes:{f}'].x).relu()
            
        gene_representation_in = self.att_in(gene_representation)
        gene_representation_att = {}
        for edge_type in data['genes'].edges:
            gene_representation_att[edge_type]  = self.gene_network_layers[edge_type](gene_representation_in,
                                                                                      data['genes',edge_type,'genes'].edge_index)
        gene_representation_att_sum = sum(gene_representation_att.values())            
        gene_representation = self.att_out(gene_representation_att_sum) + gene_representation_in
        ox = data['genes'].x    
        data['genes'].x = gene_representation
        
        tissue_representations = {}
        symptom_representations = {}

        considered_expressions = ['mouse', 'atlas', 'tabula_muris', 'tabula_muris_sapiens', 'tabula_muris_senis']
        ot = {}
        for g in considered_expressions:
            ot[g] = data[f'tissues:{g}'].x
        
        for g in considered_expressions:
            data[f'tissues:{g}'].x = self.gene_tissue_layers[g](x_dict = data.x_dict,
                                                                edge_index_dict = data.edge_index_dict)[f'tissues:{g}'].relu()
        for g in considered_expressions:
            symptom_representations[g] = self.tissue_symptom_layers[g](x_dict = data.x_dict,
                                                                       edge_index_dict = data.edge_index_dict)['symptoms'].relu()
        for g in considered_expressions:
            data[f'tissues:{g}'].x = ot[g]
        data['genes'].x = ox
        
        # import pdb
        # pdb.set_trace()
        
        tissue_symptom_contribution = sum(symptom_representations.values())

        return self.classifier(gene_representation*tissue_symptom_contribution) + lr_X
        