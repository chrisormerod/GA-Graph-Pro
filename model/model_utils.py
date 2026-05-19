#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 06:06:04 2025

@author: cormerod
"""
import pandas as pd
from .nnlinear import NNLinear
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def test_feature(target_data: pd.DataFrame,
                 feature: pd.DataFrame,
                 epochs: int = 50,
                 lr: float = 1e-3,
                 drop: float = 0.1) -> pd.DataFrame:
    """
    Tests the predictive power of features on target data using neural networks.
    
    Parameters:
    -----------
    target_data : pd.DataFrame
        DataFrame containing target variables (symptoms) as columns
    feature : pd.DataFrame
        DataFrame containing feature variables
    epochs : int, default=50
        Number of training epochs for each neural network
    lr : float, default=0.001
        Learning rate for the optimizer
    drop : float, default=0.1
        Dropout rate for regularization in the neural network
        
    Returns:
    --------
    pd.DataFrame
        Results of model performance for each target variable
    """
    # Align features with target data using index
    feature = feature.loc[target_data.index]
    # Get the dimension of feature space
    feature_dim = len(feature.columns)
    
    # Split data into training (80%) and rest (20%) sets
    train_targets, rest_targets, train_features, rest_features = train_test_split(target_data, 
                                                                                  feature, 
                                                                                  test_size=0.2,
                                                                                  random_state=42)
    
    # Further split rest data into dev (10%) and test (10%) sets
    dev_targets, test_targets, dev_features, test_features = train_test_split(rest_targets, 
                                                                              rest_features, 
                                                                              test_size=0.5,
                                                                              random_state=42)
    
    # Store results for each target variable
    rows = []
    # Iterate through each symptom/target variable with progress bar
    for symptom in tqdm(target_data.columns, total=len(target_data.columns)):
        # Initialize neural network model
        model = NNLinear(dim=feature_dim,
                         drop=drop)
        
        # Train the model on current symptom
        model.fit(train_features, train_targets[symptom],
                  dev_features, dev_targets[symptom], 
                  epochs=epochs,
                  lr=lr)
        
        # Evaluate model on test set and store results
        rows.append(model.test(test_features, test_targets[symptom], set_name="test"))
    
    # Combine all results into a DataFrame
    results = pd.DataFrame(rows)
    return results
