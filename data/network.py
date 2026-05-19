#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 08:21:25 2025

@author: cormerod
"""

import networkx as nx
from data.get_data import get_coprecipitation

def form_network():
    coprecipitation_edges_df = get_coprecipitation()
    coprecipitation_edges_df_swap = get_coprecipitation()
    coprecipitation_edges_df_swap.columns = ['gene_b','gene_a']
    df = pd.concat([coprecipitation_edges_df, coprecipitation_edges_df_swap])
    df = df.drop_duplicates()
    G = nx.Graph()
    G.add_edges_from(zip(coprecipitation_edges_df['gene_a'],coprecipitation_edges_df['gene_b']))
    G = nx.subgraph(G, list(nx.connected_components(G))[0])
    return G

def get_statistics():
    G = form_network()
    deg_cent = pd.Series(nx.centrality.degree_centrality(G), name = "degree centrality")
