#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:06:47 2025

@author: cormerod
"""
from torch_geometric.data import HeteroData
import torch
import pandas as pd
from .graphical_nn import GeometricNeuralNetwork
from utils.data_utils import (create_kfold_splits, 
                              get_geometric_data_split)
from utils.metrics import get_row
from utils.results import ResultLog
import os
from tqdm import tqdm

class RunGeneric():
    
    
    def __init__(self, 
                 data : HeteroData,
                 target_data : pd.DataFrame,
                 hidden : int = 128,
                 dropout : float = 0.4,
                 learning_rate : float = 1e-3,
                 epochs : int = 20,
                 logging_epochs : int = 5,
                 run_name : str = "initial"):
        self.splits = create_kfold_splits(target_data.index)
        
        self.data = data
        self.target_data = target_data
        self.logging_epochs = logging_epochs
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden = hidden
        self.dropout = dropout
        self.run_name = run_name
        try:
            os.mkdir(f"results/{run_name}")
        except:
            pass
        
    def run_model(self, n_trails = 10):
        for split_num, split in enumerate(self.splits):
            self.model = GeometricNeuralNetwork(self.data, 
                                                hidden = self.hidden,
                                                dropout = self.dropout)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, total_iters=len(self.target_data.columns)*self.epochs)
            # self.results = ResultLog(target_data = self.target_data,
            #                          config = self.config(),
            #                          logging_epochs = self.logging_epochs,
            #                          name = self.run_name,
            #                          split = split_num)
            self.run_split(split, split_num)
            
    def run_split(self, split, split_num):
        for e in range(self.epochs):
            self.results = ResultLog(self.target_data,
                                     {},
                                     self.logging_epochs,
                                     self.run_name,
                                     split_num)
            qbar = tqdm(self.target_data.columns)
            for symptom in qbar:
                # if not self.results.check_complete(symptom):
                train_data, dev_data, test_data = get_geometric_data_split(self.data, split, symptom)
                symptom_index = list(self.target_data.columns).index(symptom)
                train_y = train_data['genes:targets'].x[:,symptom_index]
                dev_y = dev_data['genes:targets'].x[:,symptom_index]
                test_y = test_data['genes:targets'].x[:,symptom_index]
                self.train_model(symptom, train_data, train_y, dev_data, dev_y, test_data, test_y)
                qbar.desc = self.results.get_desc()
            self.full_evaluate(e, split, split_num)
                
    def train_model(self, 
                     symptom,
                     train_data, 
                     train_y,
                     dev_data, 
                     dev_y,
                     test_data, 
                     test_y):
        for _ in range(self.logging_epochs):
            self.training_step(train_data, train_y)

        dev_metrics = self.evaluate(symptom, train_data, train_y, dev_data, dev_y)
        dev_metrics.update(self.evaluate(symptom, train_data, train_y, test_data, test_y,set_name="test"))
        self.results.finalize(symptom, dev_metrics)       
  
        
    def training_step(self, 
                      train_data : HeteroData, 
                      train_y : torch.Tensor, 
                      class_weight = None):
        """
        Performs a single training step for the model.
        
        This method executes one forward and backward pass through the neural network,
        calculates the loss, and updates the model parameters using the provided optimizer.
        
        Args:
            model: The neural network model being trained
            train_data: Input features/data for training
            train_y: Target labels for the training data
            optimizer: The optimization algorithm (e.g., SGD, Adam) for updating weights
            scheduler: Learning rate scheduler for adjusting the learning rate
            class_weight: Optional tensor of weights for handling class imbalance
                         (default: None)
        
        Returns:
            None. The model is updated in-place through the optimizer.
        """        
        # Reset gradients to zero before the forward pass
        

        self.model.zero_grad(set_to_none=True)
        for v in train_data.x_dict.values():
            v.grad=None
        self.model.zero_grad()
        
        # Forward pass: Get model predictions
        pred_y = self.model.forward(train_data)
        
        # Calculate cross-entropy loss, applying class weights if provided
        loss = torch.nn.functional.cross_entropy(pred_y, train_y.long())

        # Backward pass: Calculate gradients
        loss.backward()
        
        # Update model parameters based on gradients
        self.optimizer.step()
        
        # Update learning rate according to scheduler
        self.scheduler.step()
        return str(float(loss))
    
        
        
    def evaluate(self, symptom, train_data, train_y, dev_data, dev_y, set_name = "dev"):
        # import pdb
        # pdb.set_trace()
        metrics = {"symptom":symptom}
        with torch.no_grad():
            dev_pred = self.model(dev_data)
        train_pred_targets = [int(x) for x in (dev_pred[:len(train_y)].argmax(-1)).cpu().numpy()]
        train_true_targets = [int(x) for x in train_y]
        
        dev_pred_targets = [int(x) for x in (dev_pred[len(train_y):].argmax(-1)).cpu().numpy()]
        dev_true_targets = [int(x) for x in dev_y[len(train_y):]]
        
        metrics.update(get_row("", train_true_targets, train_pred_targets, set_name = "train"))
        metrics.update(get_row("", dev_true_targets, dev_pred_targets, set_name = set_name))
        
        return metrics    
    
    def full_evaluate(self, epoch : int, split, split_num):
        print("Evaluating")
        rows = []
        
        qbar = tqdm(self.target_data.columns)
        for symptom in qbar:
            train_data, dev_data, test_data = get_geometric_data_split(self.data, split, symptom)
            symptom_index = list(self.target_data.columns).index(symptom)
            train_y = train_data['genes:targets'].x[:,symptom_index]
            dev_y = dev_data['genes:targets'].x[:,symptom_index]
            test_y = test_data['genes:targets'].x[:,symptom_index]
            metrics = self.evaluate(symptom, train_data, train_y, dev_data, dev_y)
            metrics.update(self.evaluate(symptom, train_data, train_y, test_data, test_y, set_name="test"))
            rows.append(metrics)
        import pdb
        pdb.set_trace()
        
    
    