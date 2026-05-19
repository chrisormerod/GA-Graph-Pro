#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:02:34 2025

@author: cormerod
"""
from typing import List, Union, Tuple
from torch_geometric.data import HeteroData
import torch

class Gene:
    
    def __init__(self, gene):
        self.name = gene
        self.tissues = []
        
    def __repr__(self):
        return f"(Gene:{self.name}:{self.tissues})"

class Symptom:
    
    def __init__(self, symptom):
        self.name = symptom
        self.tissues = []
            
    def __repr__(self):
        return f"(Symptom:{self.name}:{self.tissues})"

class ExpressionData:
    
    def __init__(self):
        self.expressions = {}
        self.genes = {}
        self.symptoms = {}
        
    def add_expression(self, group):
        self.expressions[group] = Expression(group)

    def add_tissue(self, 
                   group, 
                   name):
        self.expressions[group].add_tissue(name)
        
    def to_json(self):
        pass

    def merge_with(self, 
                   data : HeteroData):
        for group,exp in self.expressions.items():
            
            data.expressions.append(group)
            data[f'tissues:{group}'].x = torch.zeros(size = (len(exp),data.hidden))
            data.labels[f'tissues:{group}'] = sorted(list(exp.tissues))
            data[f'tissues:{group}'].index_mapping = {t:k for k,t in enumerate(data.labels[f'tissues:{group}'])}
            
            gene_tissue_edge_lists = {'genes':[],
                                      'tissues':[]}
            
            for gene_name, gene in self.genes.items():
                if gene_name in data.labels['genes']:
                    idx_g = data['genes'].index_mapping[gene_name]
                    for t in gene.tissues:
                        if t.parent.group == exp.group:
                            idx_t = data[f'tissues:{group}'].index_mapping[t.name]
                            gene_tissue_edge_lists['genes'].append(idx_g)
                            gene_tissue_edge_lists['tissues'].append(idx_t)
            data["genes","",f'tissues:{group}'].edge_index = torch.LongTensor([gene_tissue_edge_lists['genes'],
                                                                            gene_tissue_edge_lists['tissues']])
            data[f'tissues:{group}',"","genes"].edge_index = torch.LongTensor([gene_tissue_edge_lists['tissues'],
                                                                            gene_tissue_edge_lists['genes']])            

            symptom_tissue_edge_lists = {'symptoms':[],
                                         'tissues':[]}
            
            for symptom_name, symptom in self.symptoms.items():
                if symptom_name in data.labels['symptoms']:
                    idx_s = data['symptoms'].index_mapping[symptom_name]
                    for t in symptom.tissues:
                        if t.parent.group == exp.group:
                            idx_t = data[f'tissues:{group}'].index_mapping[t.name]
                            symptom_tissue_edge_lists['symptoms'].append(idx_s)
                            symptom_tissue_edge_lists['tissues'].append(idx_t)
            data["symptoms","",f'tissues:{group}'].edge_index = torch.LongTensor([symptom_tissue_edge_lists['symptoms'],
                                                                               symptom_tissue_edge_lists['tissues']])
            data[f'tissues:{group}',"","symptoms"].edge_index = torch.LongTensor([symptom_tissue_edge_lists['tissues'],
                                                                               symptom_tissue_edge_lists['symptoms']])            
        
    def __contains__(self, item : Tuple):
        group = item[0]
        tissue = item[1]

    def connect_symptom(self, group, tissue, symptom):
        if tissue not in self.expressions[group]:
            self.add_tissue(group, tissue)
        if symptom in self.symptoms:
            s = self.symptoms[symptom]
        else:
            self.symptoms[symptom] = Symptom(symptom)
            s = self.symptoms[symptom]
        self.expressions[group].connect_symptom(tissue, s)
            
    def connect_gene(self, group, tissue, gene, level = 1):
        if tissue not in self.expressions[group]:
            self.add_tissue(group, tissue)
        if gene in self.genes:
            g = self.genes[gene]
        else:
            self.genes[gene] = Gene(gene)
            g = self.genes[gene]
        self.expressions[group].connect_gene(tissue, g, level)
        
    def to_dict(self):
        return None
    
    def trace_symptom(self, symptom, threshold = 0):
        s = self.symptoms[symptom]
        genes = set()
        for t in s.tissues:
            for g in t.genes:
                if t.genes[g] > threshold:
                    genes.add(g)
        return list(genes)

class Expression:
    
    def __init__(self, group):
        self.group = group
        self.tissues = {}
        
    def add_tissue(self, 
                   name : str):
        if name not in self:
            self.tissues[name] = Tissue(self, name)

    def to_dict(self):
        return {"group":self.group,
                "tissues":{k:t.to_dict() for k,t in self.tissues.items()}}
    
    def __contains__(self, tissue):
        return tissue in self.tissues

    def connect_symptom(self, 
                        tissue : str, 
                        s : Symptom):
        self.tissues[tissue].connect_symptom(s)
        
    def connect_gene(self, 
                     tissue : str, 
                     g : Gene, 
                     level : float= 1):
        self.tissues[tissue].connect_gene(g, level)
    
    def __len__(self):
        return len(self.tissues)

class Tissue:
    
    def __init__(self,
                 group : Expression,
                 name : str):
        self.parent = group
        self.name = name
        self.genes = {}
        self.symptoms = {}
        self.equivalencies = []

    def to_dict(self):
        return {"name":self.name,
                "genes":self.genes,
                "symptoms":self.symptoms,
                "equivalencies":self.equivalencies}
        
    def __repr__(self):
        return f"[{self.name}]"
    
    def connect_symptom(self, 
                        s : Symptom):
        self.symptoms[s.name] = 1
        s.tissues.append(self)
    
    def connect_gene(self, 
                     g : Gene,
                     level = 1):
        self.genes[g.name] = level
        g.tissues.append(self)
        