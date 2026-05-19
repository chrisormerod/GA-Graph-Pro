#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:22:24 2024

@author: cormerod
"""

import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from utils.metrics import get_row


class NNLinear(nn.Module):
    
    # Initialize the model with the following parameters:
    def __init__(self, 
                 dim: int,  # input dimensionality of the data
                 drop: float = 0.1,  # dropout rate for regularization
                 label_smoothing: float = 0.1):  # smoothing factor for labels (default=0.1)
        """
        Initialize the model with the given parameters.

        Args:
            dim (int): Input dimensionality of the data.
            drop (float, optional): Dropout rate for regularization. Defaults to 0.1.
            label_smoothing (float, optional): Smoothing factor for labels. Defaults to 0.1.
        """
        super().__init__()  # Call the parent class's constructor
        self.label_smoothing = label_smoothing  # Store the smoothing factor for labels
        self.drop = nn.Dropout(drop)  # Initialize dropout layer with given rate
        
        # Create a linear classifier with 2 output neurons (binary classification)
        self.classifier = nn.Linear(in_features=dim, out_features=2)

    # Define the forward pass through the network
    def forward(self, X: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            X (torch.Tensor): Input tensor to be processed

        Returns:
            torch.Tensor: Output of the neural network
        """
        # Apply dropout to the input data
        X = self.drop(X)
        
        # Pass the output through the linear classifier
        return self.classifier(X)

    # Method for initializing and configuring the model for training
    def initialize(self, 
                  y_train: pd.Series,  # Training labels
                  lr: float = 1e-4,  # Learning rate (default=0.001)
                  epochs: int = 200):  # Number of training epochs (default=200)
        """
        Initialize and configure the model for training.

        Args:
            y_train (pd.Series): Training labels.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            epochs (int, optional): Number of training epochs. Defaults to 200.
        """
        # Set the model to train mode
        self.train()
        
        try:
            # Compute class weights for balanced classification
            self.class_weight = torch.Tensor(compute_class_weight("balanced", 
                                                                  classes=np.array([1,0]), 
                                                                  y=y_train))
        except Exception as e:
            # Default to equal class weights if computation fails
            self.class_weight = torch.Tensor([1, 1])
        
        # Create an Adam optimizer with the given learning rate and parameters
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
        # Set up a linear learning rate scheduler for the optimizer
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                           start_factor=1, 
                                                           end_factor=0.01,  # Changed from 1 to 0.01
                                                           total_iters=epochs)  # Number of training epochs    
    def fit(self, 
            X_train : pd.DataFrame, 
            y_train : pd.Series, 
            X_dev : pd.DataFrame = None,
            y_dev : pd.DataFrame = None,
            epochs : int = 200,
            lr : float = 1e-4,
            early_stopping = True):
        self.initialize(y_train)
        X_train_tensor = torch.Tensor(X_train.values)
        y_train_tensor = torch.LongTensor(y_train)
        if X_dev is not None:
            X_dev_tensor = torch.Tensor(X_dev.values)
            y_dev_tensor = torch.LongTensor(y_dev)
        
        X_train_tensor,y_train_tensor = X_train_tensor.to(self.device()), y_train_tensor.to(self.device())
        if X_dev is not None:
            X_dev_tensor,y_dev_tensor = X_dev_tensor.to(self.device()), y_dev_tensor.to(self.device())
            best_kappa = 0
            best_weights = self.state_dict()            
                
        self.class_weight = self.class_weight.to(self.device())
        for _ in range(epochs):
            self.training_step(X_train_tensor, 
                               y_train_tensor)
            if X_dev is not None:
                dev_metrics = self.test(X_dev,
                                        y_dev, set_name = "dev")
                if dev_metrics['kappa_dev'] >= best_kappa:
                    best_weights = self.state_dict()
        if X_dev is not None:
            self.load_state_dict(best_weights)
        X_train_tensor,y_train_tensor = X_train_tensor.cpu(), y_train_tensor.cpu()
        self.class_weight = self.class_weight.cpu()
        
    
    def training_step(self, 
                      X : torch.Tensor,
                      y : torch.LongTensor):
        self.train()
        self.optimizer.zero_grad()
        y_pred = self.forward(X)
        loss = nn.functional.cross_entropy(y_pred, y, 
                                           weight=self.class_weight,
                                           label_smoothing=self.label_smoothing)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm = 1)
        self.optimizer.step()
        self.scheduler.step()
        
    def device(self) -> torch.device:
        return self.classifier.weight.device
        
    def test(self, 
             X_test : pd.DataFrame,
             y_test : pd.Series,
             symptom : str = "",
             set_name = ""):

        X_tensor = torch.Tensor(X_test.values)
        self.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device())
            y_pred_logproba = self.forward(X_tensor)
            y_pred = y_pred_logproba.argmax(-1)
            X_tensor = X_tensor.cpu()
        return get_row(symptom, y_test, y_pred.cpu(), set_name=set_name)
    
    