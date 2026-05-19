#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:18:29 2025

@author: cormerod
"""
import pandas as pd
from .download_raw import (curated_correspondences_file,
                           curated_correspondences_tabula_file,
                           human_atlas_levels_raw,
                           normalize_tissue_raw, 
                           expression_alliance_raw,
                           mouse_mapping_raw)
from .expression import (ExpressionData,
                         Expression,
                         Tissue)

mouse_expression_file = "data/generated/mouse_expression.feather"
atlas_expression_file = "data/generated/atlas_expression.feather"
normal_atlas_expression_file = "data/generated/normal_atlas_expression.feather"
tabula_muris_expression_file = "data/generated/tissue/TabulaMuris-Binary.csv"
tabula_muris_sapiens_expression_file = "data/generated/tissue/TabulaMurisSapiens-Binary.csv"
tabula_muris_senis_expression_file = "data/generated/tissue/TabulaMurisSenis-Binary.csv"


def build_expression_data():
    print("Building Expression Data: ", end="")
    data = ExpressionData()
    
    data.add_expression("mouse")
    data.add_expression("atlas")
    data.add_expression("tabula_muris")
    data.add_expression("tabula_muris_sapiens")
    data.add_expression("tabula_muris_senis")
    
    connect_symptoms(data)
    connect_genes(data)
    print("Complete")
    return data

def connect_symptoms(data : ExpressionData):
    correspondence_data = pd.read_excel(curated_correspondences_file)
    
    for idx, row in correspondence_data.iterrows():
        symptom = row['hpo_id']
        
        atlas_tissues = [x.strip() for x in str(row['adult_human_atlas']).split(";") if x not in {"nan",""}]
        for t in atlas_tissues:
            data.connect_symptom("atlas", t, symptom)
            
        mouse_tissues = [x for x in str(row['emb_mouse_tissues']).split(";") + str(row['adult_mouse_tissues']).split(";") if x != "nan"]
        for t in mouse_tissues:
            data.connect_symptom("mouse", t, symptom)
        
    correspondence_data = pd.read_excel(curated_correspondences_tabula_file)
    for idx, row in correspondence_data.iterrows():
        symptom = row['hpo_id']
        
        tabula_muris_tissues = [x.strip() for x in str(row['Tabula Muris']).split(";") if x not in {"nan",""}]
        for t in tabula_muris_tissues:
            data.connect_symptom("tabula_muris", t, symptom)
        
        tabula_muris_senis_tissues = [x.strip() for x in str(row['Tabula Muris Senis']).split(";") if x not in {"nan",""}]
        for t in tabula_muris_senis_tissues:
            data.connect_symptom("tabula_muris_senis", t, symptom)
            
        tabula_muris_sapiens_tissues = [x.strip() for x in str(row['Tabula Sapiens']).split(";") if x not in {"nan",""}]
        for t in tabula_muris_sapiens_tissues:
            data.connect_symptom("tabula_muris_sapiens", t, symptom)

def connect_genes(data : ExpressionData):
    connect_atlas(data)
    connect_mouse(data)
    connect_tabula_muris(data)
    connect_tabula_muris_sapiens(data)
    connect_tabula_muris_senis(data)
    
def connect_atlas(data : ExpressionData):
    atlas_expression_data = pd.read_csv(human_atlas_levels_raw,sep="\t")
    atlas_expression_data = atlas_expression_data[atlas_expression_data['nTPM'] > 0]
    normalized = pd.read_csv(normalize_tissue_raw, sep="\t")
    normalized = normalized[normalized['Level'].map(lambda x:x in {"High","Medium"})]
    normalized = normalized[normalized['Reliability'].map(lambda x:x in {"Approved","Enhanced"})]
    approved_edges = set(normalized.apply(lambda x:(x['Tissue'].lower(),x['Gene name']),1))
    
    for idx, row in atlas_expression_data.iterrows():
        gene = row['Gene name']
        tissue = row['Tissue']
        level = row['nTPM']
        if (tissue, gene) in approved_edges:
            data.connect_gene("atlas", tissue, gene, level)

def get_raw_mouse_expression():   
    mouse_expression_data = pd.read_csv(expression_alliance_raw, sep="\t", skiprows=14)
    mouse_expression_data['GeneSymbol'] = mouse_expression_data['GeneSymbol'].map(lambda x:x.upper())
    mouse_expression_data = mouse_expression_data.drop_duplicates(['GeneSymbol', "AnatomyTermName"])
    return mouse_expression_data
    
def connect_mouse(data):
    mouse_expression_data = get_raw_mouse_expression()
    
    for idx, row in mouse_expression_data.iterrows():
        gene = row['GeneSymbol']
        tissue = row['AnatomyTermName']
        
        data.connect_gene("mouse", tissue, gene)

def make_mouse_expression():
    mouse_expression_data = get_raw_mouse_expression()
    mouse_expression_data['value'] = 1
    df = pd.pivot_table(mouse_expression_data,
                        values = "value",
                        index='GeneSymbol',
                        columns='Location',
                        fill_value=0)
    df.to_feather(mouse_expression_file)
    
def make_atlas_expression():
    atlas_expression_data = pd.read_csv(human_atlas_levels_raw,sep="\t")
    df = pd.pivot_table(atlas_expression_data,
                        values = "nTPM",
                        index='Gene name',
                        columns='Tissue',
                        fill_value=0)
    df.to_feather(atlas_expression_file)
    
def make_atlas_nromal_expression():
    normal_expression = pd.read_csv(normalize_tissue_raw,sep="\t")
    normal_expression = normal_expression[normal_expression['Level'].map(lambda x:x in {"Medium","High"})]
    normal_expression['value'] = 1
    normal_expression['Tissue:cell'] = normal_expression.apply(lambda x:f"{x['Tissue']}:{x['Cell type']}",1)
    df = pd.pivot_table(normal_expression, 
                        values = "value",
                        index = 'Gene name',
                        columns = 'Tissue:cell',
                        fill_value=0)
    df.to_feather(normal_atlas_expression_file)

def connect_tabula_muris(data):
    normal_expression = pd.read_csv(tabula_muris_expression_file)
    normal_expression = normal_expression.set_index("gene")
    result = normal_expression.stack().loc[lambda s: s == 1].index.tolist()
    for gene, tissue in result:
        data.connect_gene("tabula_muris", tissue, gene.upper())

def connect_tabula_muris_sapiens(data):
    normal_expression = pd.read_csv(tabula_muris_sapiens_expression_file)
    normal_expression = normal_expression.set_index("gene")
    result = normal_expression.stack().loc[lambda s: s == 1].index.tolist()
    for gene, tissue in result:
        data.connect_gene("tabula_muris_sapiens", tissue, gene.upper())
    
def connect_tabula_muris_senis(data):
    normal_expression = pd.read_csv(tabula_muris_senis_expression_file)
    normal_expression = normal_expression.set_index("gene")
    result = normal_expression.stack().loc[lambda s: s == 1].index.tolist()
    for gene, tissue in result:
        data.connect_gene("tabula_muris_senis", tissue, gene.upper())