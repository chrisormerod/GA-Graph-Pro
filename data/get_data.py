import pandas as pd
import numpy as np
from typing import List
from utils.data_utils import (set_index,
                              restrict_dataframe)
from .parse_raw import (full_go_file,
                        pathway_file,
                        complex_file,
                        embedding_file,
                        subcellular_location_file,
                        target_data_file,
                        mapping_uniprot_name_file)
from .parse_expressions import (mouse_expression_file,
                                atlas_expression_file,
                                normal_atlas_expression_file,
                                tabula_muris_expression_file,
                                tabula_muris_sapiens_expression_file,
                                tabula_muris_senis_expression_file)
from .parse_connections import (coprecipitation_adjacency_file,
                                metabolic_adjacency_file,
                                transcription_adjacency_file,
                                phosphorylation_adjacency_file,
                                dephosphorylation_adjacency_file)

full_go_embedding_file = "data/generated/full_go_embedding.feather"
complex_embedding_file = "data/generated/complex_embedding.feather"
pathway_embedding_file = "data/generated/pathway_embedding.feather"

from .parse_connections import (coprecipitation_file,
                                metabolic_file,
                                transcription_file,
                                phosphorylation_file,
                                dephosphorylation_file)

def get_target_data(threshold: int = 0) -> pd.DataFrame:
    """
    Loads target data from a Feather file, filters columns based on their sum values,
    and returns a DataFrame containing only those columns with sums greater than or equal to the specified threshold.

    Args:
        threshold (int): The minimum value for column sums. Defaults to 0 if not provided.
    
    Returns:
        pd.DataFrame: A filtered version of the target data, excluding columns with sum values below the threshold.
    """

    # Load the target data from a Feather file
    data = pd.read_feather(target_data_file)

    return restrict_dataframe(data, threshold)

def get_descriptions():
    with open("data/raw/hp.obo",'r') as fp:
        data = [[y for y in x.split("\n") if len(y)>0] for x in "".join(fp.readlines()).split("[Term]")[1:]]
    descriptions = {}
    for x in data:
        hpo_id = x[0][4:]
        name = x[1][6:]
        descriptions[hpo_id] = {"name":name}
    return descriptions    
    
        

def get_uniprot_gene_correspondence():
    with open(mapping_uniprot_name_file, "r") as fp:
        import json
        mapping = json.load(fp)
    return mapping

def convert_uniprot_to_genes(data : pd.DataFrame) -> pd.DataFrame:
    mapping = get_uniprot_gene_correspondence()
    new_rows = {}
    for uniprot_id, row in data.iterrows():
        if uniprot_id in mapping:
            for gene_name in mapping[uniprot_id]:
                new_rows[gene_name] = np.array(row)
    df = pd.DataFrame(new_rows, index = data.columns).transpose()
    return df
        
#%% Gene-based features

def get_embedding_genes(genes = None):
    data = pd.read_feather(embedding_file)
    df = convert_uniprot_to_genes(data)
    if genes is None:
        return df
    else:
        return set_index(df, genes)

def get_subcellular_location(genes = None, 
                             embedding : int = 0):
    data = pd.read_feather(subcellular_location_file)
    if genes is None:
        return data
    else:
        return set_index(data, genes)
    
def get_full_go_terms(genes: List[str] = None, 
                threshold : int = 0,
                embedding : int = 0):
    if embedding == 0:
        data = pd.read_feather(full_go_file)
    else:
        data = pd.read_feather(full_go_embedding_file)
    data = restrict_dataframe(data, threshold)
    if genes is None:
        return data
    else:
        return set_index(data, genes)

        
         
def get_pathways(genes: List[str] = None, 
                 threshold : int = 0,
                 embedding : int = 0):
    if embedding == 0:
        data = pd.read_feather(pathway_file)
    else:
        data = pd.read_feather(pathway_embedding_file)
    data = restrict_dataframe(data, threshold)
    df = convert_uniprot_to_genes(data)
    if genes is None:
        return restrict_dataframe(df,threshold)
    else:
        return restrict_dataframe(set_index(df, genes),threshold)
        
def get_complexes(genes: List[str] = None, 
                  threshold : int = 0,
                  embedding : int = 0):
    if embedding == 0:
        data = pd.read_feather(complex_file)
    else:
        data = pd.read_feather(complex_embedding_file)
    data = restrict_dataframe(data, threshold)
    df = convert_uniprot_to_genes(data)
    if genes is None:
        return restrict_dataframe(df,threshold)
    else:
        return restrict_dataframe(set_index(df, genes),threshold)

#%% Expression-based Features    

def get_mouse_expression(genes : List[str] = None,
                         threshold : int = 0,
                         embedding : int = 0):
    df = pd.read_feather(mouse_expression_file)
    df.columns = [f"Mouse:{c}" for c in df.columns]
    if genes is None:
        return restrict_dataframe(df,threshold)
    else:
        return restrict_dataframe(set_index(df, genes),threshold)
    
def get_atlas_expression(genes : List[str] = None,
                         threshold : int = 0,
                         embedding : int = 0):
    df = pd.read_feather(atlas_expression_file)
    df.columns = [f"Atlas:{c}" for c in df.columns]
    if genes is None:
        return restrict_dataframe(df,threshold)
    else:
        return restrict_dataframe(set_index(df, genes),threshold)

def get_normal_atlas_expression(genes : List[str] = None,
                                threshold : int = 0,
                                embedding : int = 0):
    df = pd.read_feather(normal_atlas_expression_file)
    df.columns = [f"AtlasNormal:{c}" for c in df.columns]
    if genes is None:
        return restrict_dataframe(df,threshold)
    else:
        return restrict_dataframe(set_index(df, genes),threshold)

def get_tabula_muris_expression(genes : List[str] = None,
                                threshold : int = 0,
                                embedding : int = 0):
    df = pd.read_csv(tabula_muris_expression_file)
    df = df.set_index('gene')
    df.columns = [f"TabulaMuris:{c}" for c in df.columns]
    if genes is None:
        return restrict_dataframe(df,threshold)
    else:
        return restrict_dataframe(set_index(df, genes),threshold)
    
def get_tabula_muris_senis_expression(genes : List[str] = None,
                                      threshold : int = 0,
                                      embedding : int = 0):
    df = pd.read_csv(tabula_muris_senis_expression_file)
    df = df.set_index('gene')
    df.columns = [f"TabulaMurisSenis:{c}" for c in df.columns]
    if genes is None:
        return restrict_dataframe(df,threshold)
    else:
        return restrict_dataframe(set_index(df, genes),threshold)
    

def get_tabula_muris_sapiens_expression(genes : List[str] = None,
                                        threshold : int = 0,
                                        embedding : int = 0):
    df = pd.read_csv(tabula_muris_sapiens_expression_file)
    df = df.set_index('gene')
    df.columns = [f"TabulaMurisSapiens:{c}" for c in df.columns]
    if genes is None:
        return restrict_dataframe(df,threshold)
    else:
        return restrict_dataframe(set_index(df, genes),threshold)
    
def get_coprecipitation(genes : List[str] = None):
    data = pd.read_feather(coprecipitation_file)
    if genes is not None:
        df = data[data.apply(lambda x:(x['gene_a'] in genes) and (x['gene_b'] in genes) ,1)]
    else: 
        df = data
    return df


def get_metabolic(genes : List[str] = None):
    data = pd.read_feather(metabolic_file)
    if genes is not None:
        df = data[data.apply(lambda x:(x['gene_a'] in genes) and (x['gene_b'] in genes) ,1)]
    else: 
        df = data
    return df


def get_transcription(genes : List[str] = None):
    data = pd.read_feather(transcription_file)
    if genes is not None:
        df = data[data.apply(lambda x:(x['gene_a'] in genes) and (x['gene_b'] in genes) ,1)]
    else: 
        df = data
    return df


def get_phosphorylation(genes : List[str] = None):
    data = pd.read_feather(phosphorylation_file)
    if genes is not None:
        df = data[data.apply(lambda x:(x['gene_a'] in genes) and (x['gene_b'] in genes) ,1)]
    else: 
        df = data
    return df


def get_dephosphorylation(genes : List[str] = None):
    data = pd.read_feather(dephosphorylation_file)
    if genes is not None:
        df = data[data.apply(lambda x:(x['gene_a'] in genes) and (x['gene_b'] in genes) ,1)]
    else: 
        df = data
    return df

def get_coprecipitation_adjacency(genes : List[str] = None,
                                  threshold : int = 0):
    df = pd.read_feather(coprecipitation_adjacency_file)
    if genes is None:
        return restrict_dataframe(df,threshold)
    else:
        return restrict_dataframe(set_index(df, genes),threshold)


def get_metabolic_adjacency(genes : List[str] = None,
                                  threshold : int = 0):
    df = pd.read_feather(metabolic_adjacency_file)
    if genes is None:
        return restrict_dataframe(df,threshold)
    else:
        return restrict_dataframe(set_index(df, genes),threshold)
    

def get_transcription_adjacency(genes : List[str] = None,
                                  threshold : int = 0):
    df = pd.read_feather(transcription_adjacency_file)
    if genes is None:
        return restrict_dataframe(df,threshold)
    else:
        return restrict_dataframe(set_index(df, genes),threshold)
    
    

def get_phosphorylation_adjacency(genes : List[str] = None,
                                  threshold : int = 0):
    df = pd.read_feather(phosphorylation_adjacency_file)
    if genes is None:
        return restrict_dataframe(df,threshold)
    else:
        return restrict_dataframe(set_index(df, genes),threshold)
    
    

def get_dephosphorylation_adjacency(genes : List[str] = None,
                                  threshold : int = 0):
    df = pd.read_feather(dephosphorylation_adjacency_file)
    if genes is None:
        return restrict_dataframe(df,threshold)
    else:
        return restrict_dataframe(set_index(df, genes),threshold)