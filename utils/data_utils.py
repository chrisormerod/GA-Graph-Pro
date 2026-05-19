import pandas as pd
from torch_geometric.data import HeteroData
import numpy as np
from typing import List, Dict
from sklearn.model_selection import KFold
import numpy as np
import random
import torch
from collections import Counter

def select_features(split_number : int,
                    threshold : int = 10):
    return {k:get_features_type(split_number, k, threshold) 
            for k in ['gene','expression','network']}
 

def get_features_type(split_number : int, 
                      feature_type : str, 
                      threshold : int = 2):
    df = pd.read_csv(f"results/feature_selection_mcc_{feature_type}_{split_number}.csv")
    df = df.drop_duplicates(keep=False)
    df['features'] = df['features'].map(eval)
    df = df[df['N_dev'].map(int)>= threshold]
    df = df[df['kappa_dev'].map(float)>= 0.2]
    count = sum(df['features'].map(Counter),Counter())
    features = sorted(list(count.keys()))
    return features
    
def merge_list(L: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Recursively merges a list of pandas DataFrames using their indices.
    
    Args:
        L (List[pd.DataFrame]): A list of pandas DataFrames to be merged
        
    Returns:
        pd.DataFrame: A single DataFrame containing all merged data
        
    """
    # Base case: if only one DataFrame remains, return it
    if len(L) == 1:
        return L[0]
    else:
        # Recursive case: merge first DataFrame with result of merging remaining DataFrames
        return pd.merge(left = L[0], 
                       right = merge_list(L[1:]),   # Bug: missing argument L[1:]
                       how = "outer",               # Keep all index values from both DataFrames
                       left_index=True,             # Use index for merging
                       right_index = True)          # Use index for merging
    

def fill_df(df: pd.DataFrame, 
            index: List[str]) -> pd.DataFrame:
    """
    Fills missing values in a DataFrame with zeros and merges it back to its original index.

    Args:
        df (pd.DataFrame): The input DataFrame that may contain missing values.
        index (List[str]): A list of indices where the DataFrame is incomplete.

    Returns:
        pd.DataFrame: The completed DataFrame with all rows filled, including those from the provided indices.
    """

    # Identify the missing indices in the original DataFrame
    missing_idx = [x for x in index if x not in df.index]

    # Create a new DataFrame to hold the missing values, initialized with zeros and matching the original index
    missing_rows = pd.DataFrame(index=missing_idx, columns=df.columns).fillna(0)

    # Merge the completed rows back into the original DataFrame using its indices
    full_df = pd.concat([df, missing_rows]).loc[index]

    return full_df


def set_index(df: pd.DataFrame, 
             index: List[str]):
    """
    Reindexes a DataFrame based on a provided list of indices.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to be reindexed
    index : List[str]
        The list of indices to set as the new index of the DataFrame
        
    Returns:
    --------
    pd.DataFrame
        A new DataFrame with the specified index order
        
    Notes:
    ------
    This function performs these steps:
    1. Filters the input DataFrame to only include rows with indices present in the provided index list
    2. Calls an external function 'fill_df' to handle any missing indices 
    3. Returns the DataFrame with rows ordered according to the provided index list
    """
    # Filter to only include rows where the current index exists in the target index list
    included_indices = [idx for idx in df.index if idx in index]
    
    # Use an external function 'fill_df' to handle missing indices
    # (This function is not defined here but likely fills in missing values)
    full_df = fill_df(df.loc[included_indices], index)
    
    # Return the DataFrame with rows ordered according to the provided index list
    return full_df.loc[index]


def restrict_dataframe(df: pd.DataFrame, 
                       threshold: int = 0) -> pd.DataFrame:
    """
    Restricts a DataFrame to only include columns where the sum of values in those columns exceeds a specified threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        threshold (int, optional): The minimum total value required for a column to be included. Defaults to 0.

    Returns:
        pd.DataFrame: A new DataFrame containing only the columns that meet or exceed the specified threshold.
    """

    # Calculate the sum of all values in each column
    sum_rows = df.sum()

    # Filter out columns where the total value is less than or equal to the threshold
    cols_to_keep = [c for c, s in sum_rows.items() if s > threshold]

    # Return a new DataFrame containing only the selected columns
    return df[cols_to_keep]

    
def one_hot_dataframe(index_series: pd.Series,
                      column_series: pd.Series) -> pd.DataFrame():
    # Get unique values from both series to create row and column labels
    index = index_series.unique()
    columns = column_series.unique()
    
    # Create mapping dictionaries to convert labels to numeric indices
    # This allows faster array indexing compared to using labels directly
    idx_map = {k:v for v,k in enumerate(index)}    # Maps unique index values to integers 0,1,2,...
    col_map = {k:v for v,k in enumerate(columns)}  # Maps unique column values to integers 0,1,2,...
    
    # Initialize empty binary matrix with appropriate dimensions
    # Using numpy arrays for memory efficiency and faster operations
    data = np.zeros(shape = (len(index), len(columns)),
                    dtype=int)
    
    # Iterate through both series simultaneously using zip
    for idx, col in zip(index_series, column_series):
        # Set 1 at the intersection of current index and column
        # Using the mapped integer values for faster indexing
        data[idx_map[idx],col_map[col]] = 1
    
    # Convert numpy array to DataFrame
    # Restore original labels for both rows and columns
    df = pd.DataFrame(data, 
                      index = index, 
                      columns = columns)
    
    return df


    
def create_kfold_splits(genes, n_splits=10, seed=42, dev_prop = 0.2):
    """
    Divides a list of strings into multiple train, development, and test splits using KFold.
    
    Args:
        data (list): List of strings to split
        n_splits (int): Number of folds
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of dictionaries, each containing 'train', 'dev', and 'test' splits
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Convert data to array for indexing
    data_array = np.array(genes)
    
    # Create the splits
    all_splits = []
    
    for fold_idx, (train_dev_idx, test_idx) in enumerate(kf.split(data_array)):
        # Get test data for this fold
        test_data = data_array[test_idx].tolist()
        
        # For the remaining data (train+dev), use another fold to split into train and dev
        # We'll use 85% for train and 15% for dev from the train_dev set
        train_dev_data = data_array[train_dev_idx]
        
        # Calculate dev size (approximately 15% of the train_dev data)
        dev_size = int(len(train_dev_data) * dev_prop)
        
        # Random shuffle the train_dev data
        np.random.seed(seed + fold_idx)  # Different seed for each fold
        np.random.shuffle(train_dev_data)
        
        # Split into train and dev
        dev_data = train_dev_data[:dev_size].tolist()
        train_data = train_dev_data[dev_size:].tolist()
        
        # Store the splits
        split = {
            'train': train_data,
            'dev': dev_data,
            'test': test_data
        }
        all_splits.append(split)
    
    return all_splits


def get_geometric_data_split(data : HeteroData, 
                             split : Dict, 
                             symptom : str):
    
    train_idx_list = [data['genes'].index_mapping[g] for g in split['train']]
    dev_idx_list = [data['genes'].index_mapping[g] for g in split['train'] + split['dev']]
    test_idx_list = [data['genes'].index_mapping[g] for g in split['train'] + split['test']]
    
    train_indices = torch.LongTensor(train_idx_list)
    dev_indices = torch.LongTensor(dev_idx_list)
    test_indices = torch.LongTensor(test_idx_list)
    symptom_index = torch.LongTensor([data['symptoms'].index_mapping[symptom]])
        
    train_dict = {'genes':train_indices}
    train_dict['symptoms'] = symptom_index
    train_dict.update({f"genes:{f}":train_indices for f in data['genes'].features})
    train_dict.update({f"genes:targets":train_indices})
    
    dev_dict = {'genes':dev_indices}
    dev_dict['symptoms'] = symptom_index
    dev_dict.update({f"genes:{f}":dev_indices for f in data['genes'].features})
    dev_dict.update({f"genes:targets":dev_indices})

    
    test_dict = {'genes':test_indices}
    test_dict['symptoms'] = symptom_index
    test_dict.update({f"genes:{f}":test_indices for f in data['genes'].features})
    test_dict.update({f"genes:targets":test_indices})
    
    train, dev, test = data.subgraph(train_dict), data.subgraph(dev_dict), data.subgraph(test_dict)
    
    # import pdb
    # pdb.set_trace()
    
    return train, dev, test

def print_summary(data):
    print("Gene Features")
    for f in data['genes'].features:
        dim = data[f'genes:{f}'].x.shape[1]
        print(f"{f}: {dim}")
    
    