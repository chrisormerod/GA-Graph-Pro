from torch_geometric.data import HeteroData
import torch
import pandas as pd
from typing import List, Union
from utils.data_utils import select_features, restrict_dataframe, merge_list
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

from data.get_data import (get_coprecipitation,
                           get_metabolic,
                           get_transcription,
                           get_phosphorylation,
                           get_dephosphorylation)
from data.parse_expressions import build_expression_data

def default_edge_matrix(N):
    return torch.stack([torch.arange(0,N,1),torch.arange(0,N,1)])

def get_geometric_data(target_data : pd.DataFrame, 
                       genes : Union[List[str], None] = None,
                       split_number : int = 0,
                       selection_threshhold : int = 10,
                       hidden = 16):
    print("Building Network Data: ", end = "")

    selected_features = select_features(split_number, 
                                        threshold = selection_threshhold)
    
    data = HeteroData()
    data.hidden = hidden
    data.num_genes = len(target_data.index)
    data.num_symptoms = len(target_data.columns)
    data.num_tissues = 0
    data.labels = {}
    
    data['genes'].features = []
    data['genes'].edges = []
    
    add_protein_features(genes, data, selected_features['gene'])
    add_expression_features(genes, data, selected_features['expression'])
    add_network_features(genes, data, selected_features['network'])
    add_embedding_features(genes, data)
    
    data['genes:targets'].x = torch.Tensor(target_data.loc[genes].values)
    data['genes'].x = torch.zeros(size = (data.num_genes,hidden))
    data['symptoms'].x = torch.zeros(size = (data.num_symptoms,hidden))

    data.labels['genes'] = list(target_data.index)
    data.labels['symptoms'] = list(target_data.columns)
    
    data['genes'].index_mapping = {g:k for k,g in enumerate(data.labels['genes'])}
    data['symptoms'].index_mapping = {c:k for k,c in enumerate(data.labels['symptoms'])}

    # add_gene_features(data)
    add_expression_data(data)
    
    #networks
    add_gene_edges(data)
    
    print("Complete")
    return data

def add_protein_features(genes, data, columns):
    
    gene_feature = restrict_dataframe(merge_list([get_full_go_terms(genes, embedding= 0),
                                              get_complexes(genes, embedding=0),
                                              get_pathways(genes, embedding=0),
                                              get_subcellular_location(genes)]),1).loc[genes]
    data['genes:protein'].x = torch.Tensor(gene_feature[columns].values.astype(int))
    data['genes'].features.append("protein")
    data['genes:protein','','genes'].edge_index = default_edge_matrix(data.num_genes)
    
    
def add_expression_features(genes, data, columns):
    exp_feature = restrict_dataframe(merge_list([get_mouse_expression(genes),
                                              get_normal_atlas_expression(genes),
                                              get_tabula_muris_expression(genes),
                                              get_tabula_muris_sapiens_expression(genes),
                                              get_tabula_muris_senis_expression(genes)]),0).loc[genes]
    data['genes:expression'].x = torch.Tensor(exp_feature[columns].values.astype(int))
    data['genes'].features.append("expression")
    data['genes:expression','','genes'].edge_index = default_edge_matrix(data.num_genes)
    
def add_network_features(genes, data ,columns):
    net_feature = restrict_dataframe(merge_list([get_coprecipitation_adjacency(genes),
                                              get_metabolic_adjacency(genes),
                                              get_transcription_adjacency(genes),
                                              get_phosphorylation_adjacency(genes),
                                              get_dephosphorylation_adjacency(genes)]),0).loc[genes]
    data['genes:network'].x = torch.Tensor(net_feature[columns].values.astype(int))
    data['genes'].features.append("network")
    data['genes:network','','genes'].edge_index = default_edge_matrix(data.num_genes)
    
    # walk_feature = get_walk_data(genes)
    # data['genes:random_walk'].x = torch.Tensor(walk_feature.values)
    # data['genes'].features.append("random_walk")
    # data['genes:random_walk','','genes'].edge_index = default_edge_matrix(data.num_genes)
    
    
    
def add_embedding_features(genes, data):
    embedding = get_embedding_genes(genes)
    data['genes:embedding'].x = torch.Tensor(embedding.values.astype(float))
    data['genes'].features.append("embedding")
    data['genes:embedding','','genes'].edge_index = default_edge_matrix(data.num_genes)

def add_gene_edges(data):
    add_coprecipitation(data)
    add_metabolic(data)
    add_transcription(data)
    add_phosphorylation(data)
    add_dephosphorylation(data)
    print("Added Gene edges")
    
def add_symptom_edges(data):
    pass
    

def add_coprecipitation(data : HeteroData):
    # import pdb
    # pdb.set_trace()
    # Retrieve coprecipitation data for genes in the dataset
    data['genes'].edges.append("coprecipitation")
    coprec_df = get_coprecipitation(set(data.labels['genes']))
    
    # Initialize empty list to store gene pair edges
    edges = []
    
    # Create edges between gene pairs from coprecipitation data
    for ga, gb in zip(coprec_df['gene_a'],coprec_df['gene_b']):
        # Map gene names to their indices in the data structure
        idxa = data['genes'].index_mapping[ga]
        idxb = data['genes'].index_mapping[gb]
        
        # Add edge with sorted indices for consistency
        edges.append(tuple(sorted((idxa,idxb))))
    
    # Remove duplicate edges
    edges = list(set(edges))
    
    # Create bidirectional edges (make graph undirected)
    edges = edges + [tuple(reversed(list(x))) for x in edges]
    
    # Store edges in PyTorch format for the heterogeneous graph
    data['genes','coprecipitation','genes'].edge_index = torch.LongTensor(edges).transpose(0,1)

    

def add_transcription(data : HeteroData):
    # import pdb
    # pdb.set_trace()
    # Retrieve coprecipitation data for genes in the dataset
    data['genes'].edges.append("transcription")
    transcription_df = get_transcription(set(data.labels['genes']))
    
    # Initialize empty list to store gene pair edges
    edges = []
    
    # Create edges between gene pairs from coprecipitation data
    for ga, gb in zip(transcription_df['gene_a'],transcription_df['gene_b']):
        # Map gene names to their indices in the data structure
        idxa = data['genes'].index_mapping[ga]
        idxb = data['genes'].index_mapping[gb]
        
        # Add edge with sorted indices for consistency
        edges.append(tuple(sorted((idxa,idxb))))
    
    # Remove duplicate edges
    edges = list(set(edges))
    
    # Create bidirectional edges (make graph undirected)
    edges = edges + [tuple(reversed(list(x))) for x in edges]
    
    # Store edges in PyTorch format for the heterogeneous graph
    data['genes','transcription','genes'].edge_index = torch.LongTensor(edges).transpose(0,1)
    

def add_metabolic(data : HeteroData):
    # import pdb
    # pdb.set_trace()
    # Retrieve coprecipitation data for genes in the dataset
    data['genes'].edges.append("metabolic")
    metabolic_df = get_metabolic(set(data.labels['genes']))
    
    # Initialize empty list to store gene pair edges
    edges = []
    
    # Create edges between gene pairs from coprecipitation data
    for ga, gb in zip(metabolic_df['gene_a'],metabolic_df['gene_b']):
        # Map gene names to their indices in the data structure
        idxa = data['genes'].index_mapping[ga]
        idxb = data['genes'].index_mapping[gb]
        
        # Add edge with sorted indices for consistency
        edges.append(tuple(sorted((idxa,idxb))))
    
    # Remove duplicate edges
    edges = list(set(edges))
    
    # Create bidirectional edges (make graph undirected)
    edges = edges + [tuple(reversed(list(x))) for x in edges]
    
    # Store edges in PyTorch format for the heterogeneous graph
    data['genes','metabolic','genes'].edge_index = torch.LongTensor(edges).transpose(0,1)
    


def add_phosphorylation(data : HeteroData):
    # import pdb
    # pdb.set_trace()
    # Retrieve coprecipitation data for genes in the dataset
    data['genes'].edges.append("phosphorylation")
    phosphorylation_df = get_phosphorylation(set(data.labels['genes']))
    
    # Initialize empty list to store gene pair edges
    edges = []
    
    # Create edges between gene pairs from coprecipitation data
    for ga, gb in zip(phosphorylation_df['gene_a'],phosphorylation_df['gene_b']):
        # Map gene names to their indices in the data structure
        idxa = data['genes'].index_mapping[ga]
        idxb = data['genes'].index_mapping[gb]
        
        # Add edge with sorted indices for consistency
        edges.append(tuple(sorted((idxa,idxb))))
    
    # Remove duplicate edges
    edges = list(set(edges))
    
    # Create bidirectional edges (make graph undirected)
    edges = edges + [tuple(reversed(list(x))) for x in edges]
    
    # Store edges in PyTorch format for the heterogeneous graph
    data['genes','phosphorylation','genes'].edge_index = torch.LongTensor(edges).transpose(0,1)
    

def add_dephosphorylation(data : HeteroData):
    # import pdb
    # pdb.set_trace()
    # Retrieve coprecipitation data for genes in the dataset
    data['genes'].edges.append("dephosphorylation")
    dephosphorylation_df = get_phosphorylation(set(data.labels['genes']))
    
    # Initialize empty list to store gene pair edges
    edges = []
    
    # Create edges between gene pairs from coprecipitation data
    for ga, gb in zip(dephosphorylation_df['gene_a'],dephosphorylation_df['gene_b']):
        # Map gene names to their indices in the data structure
        idxa = data['genes'].index_mapping[ga]
        idxb = data['genes'].index_mapping[gb]
        
        # Add edge with sorted indices for consistency
        edges.append(tuple(sorted((idxa,idxb))))
    
    # Remove duplicate edges
    edges = list(set(edges))
    
    # Create bidirectional edges (make graph undirected)
    edges = edges + [tuple(reversed(list(x))) for x in edges]
    
    # Store edges in PyTorch format for the heterogeneous graph
    data['genes','dephosphorylation','genes'].edge_index = torch.LongTensor(edges).transpose(0,1)
    
    
# def add_gene_features(data : HeteroData):
#     add_embedding(data)
#     add_fgo_terms(data)
#     add_complexes(data)
#     add_pathways(data)
#     add_location(data)

def add_expression_data(data : HeteroData):
    data.expressions = []
    expression_data = build_expression_data()
    expression_data.merge_with(data)

# def add_symptom_data(data : HeteroData):
#     pass


# def add_location(data):
#     data['genes'].features.append("location")
#     feature = get_subcellular_location(data.labels['genes'])
#     data['genes:location'].x = torch.Tensor(feature.values)
#     data['genes:location',"genes"].edge_index = default_edge_matrix(data.num_genes)

# def add_embedding(data):
#     data['genes'].features.append("embedding")
#     embedding = get_embedding_genes(data.labels['genes'])
#     data['genes:embedding'].x = torch.Tensor(embedding.values)
#     data['genes:embedding',"genes"].edge_index = default_edge_matrix(data.num_genes)

# def add_fgo_terms(data, threshold = 10):
#     data['genes'].features.append("full_go")
#     feature = get_full_go_terms(data.labels['genes'], 
#                                 threshold = threshold)
#     data['genes:full_go'].x = torch.Tensor(feature.values)
#     data['genes:full_go','genes'].edge_index = default_edge_matrix(data.num_genes)
    
# def add_complexes(data, threshold = 10):
#     data['genes'].features.append("complex")
#     feature = get_complexes(data.labels['genes'], 
#                                 threshold = threshold)
#     data['genes:complex'].x = torch.Tensor(feature.values)
#     data['genes:complex','genes'].edge_index = default_edge_matrix(data.num_genes)

# def add_pathways(data, threshold = 10):
#     data['genes'].features.append("pathways")
#     feature = get_pathways(data.labels['genes'], 
#                                 threshold = threshold)
#     data['genes:pathways'].x = torch.Tensor(feature.values)
#     data['genes:pathways','genes'].edge_index = default_edge_matrix(data.num_genes)
