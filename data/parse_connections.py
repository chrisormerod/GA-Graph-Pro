#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 04:29:15 2025

@author: cormerod
"""
import pandas as pd
from .download_raw import biogrid_raw 
import re

coprecipitation_file = "data/generated/network_coprecipitation.feather"
metabolic_file = "data/generated/network_metabolic.feather"
transcription_file = "data/generated/network_transcription.feather"
phosphorylation_file = "data/generated/network_phosphorylation.feather"
dephosphorylation_file = "data/generated/network_dephosphorylation.feather"

coprecipitation_adjacency_file = "data/generated/coprecipitation_adjacency.feather"
metabolic_adjacency_file = "data/generated/metabolic_adjacency.feather"
transcription_adjacency_file = "data/generated/transcription_adjacency.feather"
phosphorylation_adjacency_file = "data/generated/phosphorylation_adjacency.feather"
dephosphorylation_adjacency_file = "data/generated/dephosphorylation_adjacency.feather"


def extract_entrez_gene(text):
    """
    Extracts the entry associated with entrez gene/locuslink from a string.
    
    Args:
        text (str): The input string containing database identifiers
        
    Returns:
        str: The extracted entrez gene/locuslink entry, or None if not found
    """
    pattern = r"entrez gene\/locuslink:([^|]+)"
    match = re.search(pattern, text)
    
    if match:
        return match.group(1)
    return None

def make_coprecipitation_entries():
    biogrid = pd.read_csv(biogrid_raw, sep="\t")
    coprecipitation_data = biogrid[biogrid['Taxid Interactor A'] == "taxid:9606"]
    gene_a = coprecipitation_data['Alt IDs Interactor A'].map(extract_entrez_gene)
    gene_b = coprecipitation_data['Alt IDs Interactor B'].map(extract_entrez_gene)
    df = pd.concat([gene_a,gene_b],axis=1)
    df.columns = ['gene_a','gene_b']
    df.to_feather(coprecipitation_file)

def make_adjacencies():
    make_adj(coprecipitation_file, "gene_a","gene_b", "cop", coprecipitation_adjacency_file, True)
    make_adj(metabolic_file, "gene_a","gene_b", "meta", metabolic_adjacency_file)
    make_adj(transcription_file, "gene_a","gene_b", "trans", transcription_adjacency_file)
    make_adj(phosphorylation_file, "gene_a","gene_b", "phos", phosphorylation_adjacency_file)
    make_adj(dephosphorylation_file, "gene_a","gene_b", "dephos", dephosphorylation_adjacency_file)
    
def make_adj(source_file, cola, colb, label, destination_file, bidirectional=False):
    df = pd.read_feather(source_file)
    if bidirectional:
        sa = df[cola].to_list()
        sb = df[colb].to_list()
        df_swap = pd.DataFrame(zip(sb,sa), columns = [cola,colb])
        df = pd.concat([df, df_swap]).drop_duplicates()
        
    df['value'] = 1
    adj = pd.pivot_table(df, index = cola, columns = colb, values = "value", fill_value=0)
    adj.columns = [f"{label}:{c}" for c in adj.columns]
    adj.to_feather(destination_file)
    
    