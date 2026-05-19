#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 12:25:45 2025

@author: cormerod
"""
from typing import Dict, List
from torch_geometric.data import HeteroData
import torch
import pandas as pd
from .graphical_nn import GeometricNeuralNetwork
from utils.data_utils import (create_kfold_splits, 
                              get_geometric_data_split)
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from utils.metrics import get_row
from sklearn.utils import compute_class_weight
import numpy as np
import optuna
from utils.results import ResultLog
from data import hpo_descriptions
import os
import json
import joblib

class State:
    
    def __init__(self, 
                 model : GeometricNeuralNetwork, 
                 dev_metrics : Dict):
        self.state_dict = model.state_dict()
        self.dev_metrics = dev_metrics
        
    def update(self, 
               model : GeometricNeuralNetwork, 
               dev_metrics : Dict,
               best_metric = "kappa_dev"):
        
        
        if best_metric not in dev_metrics:
            best_metric = 'kappa_train'
        
        if self.dev_metrics[best_metric] < dev_metrics[best_metric]:
            self.dev_metrics = dev_metrics
            self.state_dict = model.state_dict()
            return True
        else:
            return False
        
    def restore(self, 
                model : GeometricNeuralNetwork):
        model.load_state_dict(self.state_dict)
        return model

class Run:
    
    def __init__(self, 
                 data : HeteroData,
                 target_data : pd.DataFrame,
                 hidden : int = 128,
                 dropout : float = 0.4,
                 learning_rate : float = 1e-3,
                 epochs : int = 400,
                 logging_epochs : int = 20,
                 n_trials = 1,
                 run_name : str = "initial",
                 dev_prop = 0.1):
        self.splits = create_kfold_splits(target_data.index, dev_prop = dev_prop)
        self.target_data = target_data
        self.data = data
        self.hidden = hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.logging_epochs = logging_epochs
        self.use_cuda = torch.cuda.is_available()
        self.run_name = run_name
        try:
            os.makedirs(f"results/{self.run_name}", exist_ok=True)
            os.makedirs(f"results/{self.run_name}/dev", exist_ok=True)
            os.makedirs(f"results/{self.run_name}/test", exist_ok=True)
            os.makedirs(f"results/{self.run_name}/hyper", exist_ok=True)
            os.makedirs(f"results/{self.run_name}/weights", exist_ok=True)
            os.makedirs(f"results/{self.run_name}/linear", exist_ok=True)
        except:
            print("Logging to existing directory")
        with open(f"results/{self.run_name}/config.json","w") as fp:
            json.dump(self.config(), fp, indent = 2)

    def config(self):
        return {"run_name": self.run_name,
                "hidden":self.hidden,
                "learning_rate":self.learning_rate,
                "epochs":self.epochs,
                "dropout":self.dropout}

    def run_model(self, n_trials = 10):
        for split_num, split in enumerate(self.splits):
            self.results = ResultLog(target_data = self.target_data,
                                     config = self.config(),
                                     logging_epochs = self.logging_epochs,
                                     name = self.run_name,
                                     split_num = split_num)
            self.run_split(split = split, 
                           n_trials = n_trials,
                           split_num = split_num)
    
    def run_split(self, 
                  split : Dict, 
                  n_trials : int = 10, 
                  split_num :int = 0) -> pd.DataFrame:
        for symptom in self.target_data.columns:
            try: 
                description = hpo_descriptions[symptom]['name']
            except:
                description = "HPO-id not found"
            if not self.results.check_complete(symptom):
                print(f"Running : {symptom} : {description}")
                if n_trials > 1:
                    hyper_parameter_rows, dev_outputs, test_outputs = self.hyper_parameter_run(symptom, 
                                                                                               description,
                                                                                               split,
                                                                                               n_trials)
                else:
                    hyper_parameter_rows, dev_outputs, test_outputs = [], [], []
                status, dev_output, test_output = self.run_symptom(symptom = symptom,
                                                                   description = description,
                                                                   data = self.data,
                                                                   split = split,
                                                                   hidden = self.hidden,
                                                                   dropout = self.dropout,
                                                                   learning_rate = self.learning_rate,
                                                                   epochs = self.epochs,
                                                                   split_num = split_num)
                status.update({"symptom": symptom, 
                               "name": description,
                               "drop":self.dropout, 
                               "hidden":self.hidden, 
                               "epochs":self.epochs, 
                               "lr":self.learning_rate}) 
                
                hyper_parameter_rows.append(status)
                dev_outputs.append(dev_output)
                test_outputs.append(test_output)
                self.determine_and_save(symptom, 
                                        split_num, 
                                        hyper_parameter_rows, 
                                        dev_outputs, 
                                        test_outputs)

    def determine_and_save(self, 
                           symptom : str,
                           split_num : int,
                           hyper_parameter_rows : List,
                           dev_outputs : List,
                           test_outputs : List):
    
        if len(hyper_parameter_rows) > 1:
            pd.DataFrame(hyper_parameter_rows).to_csv(f"results/{self.run_name}/hyper/search_{split_num}_{symptom}.csv")
            index = np.argmax([x['kappa_dev:nn'] for x in hyper_parameter_rows])
        else:
            index = 0
        self.results.finalize(symptom, hyper_parameter_rows[index])
        dev_outputs[index].to_csv(f"results/{self.run_name}/dev/dev_output_{split_num}_{symptom}.csv")
        test_outputs[index].to_csv(f"results/{self.run_name}/test/test_output_{split_num}_{symptom}.csv")
            
    def hyper_parameter_run(self, 
                            symptom : str,
                            description : str,
                            split : Dict,
                            n_trials : int):
        hyper_parameter_rows = []
        dev_outputs = []
        test_outputs = []
        def opt_fun(trial : optuna.Trial):
            dropout = trial.suggest_float("dr", 0, 0.3)

            epochs = trial.suggest_int("epochs", 100, 1000)
            lr = trial.suggest_float("lr", 1e-4, 1e-3)
            
            status, dev_output, test_output = self.run_symptom(symptom = symptom,
                                                               description = description,
                                                               data = self.data,
                                                               split = split,
                                                               hidden = self.hidden,
                                                               dropout = dropout,
                                                               learning_rate = lr,
                                                               epochs = epochs)
            status.update({"symptom": symptom, 
                           "name":description,
                           "drop":dropout, 
                           "hidden":self.hidden, 
                           "epochs":epochs, 
                           "lr":lr}) 
            
            hyper_parameter_rows.append(status)
            dev_outputs.append(dev_output)
            test_outputs.append(test_output)
            if 'kappa_dev:nn' in status:
                return status['kappa_dev:nn']
            else:
                return 0.0
                
        study = optuna.create_study(direction="maximize")
        study.optimize(opt_fun, n_trials=n_trials-1)
        return hyper_parameter_rows, dev_outputs, test_outputs
        
    def run_symptom(self, 
                    symptom : str,
                    description : str,
                    data : HeteroData,
                    split : Dict, 
                    hidden : int,
                    dropout : float,
                    learning_rate : float,
                    epochs : int,
                    split_num : int):
        model = GeometricNeuralNetwork(data = data, 
                                       hidden = hidden,
                                       dropout = dropout)        
        symptom_index = list(self.target_data.columns).index(symptom)
        train_data, dev_data, test_data = get_geometric_data_split(data = data, 
                                                                   split = split, 
                                                                   symptom = symptom)
        train_y = train_data['genes:targets'].x[:,symptom_index]
        dev_y = dev_data['genes:targets'].x[:,symptom_index]
        test_y = test_data['genes:targets'].x[:,symptom_index]
        if self.use_cuda:
            model = model.cuda()
            train_data, dev_data, test_data = train_data.cuda(), dev_data.cuda(), test_data.cuda()
            train_y, dev_y, test_y = train_y.cuda(), dev_y.cuda(), test_y.cuda()
        test_metrics = self.train_model(symptom = symptom,
                                        description = description,
                                        model = model, 
                                        train_data = train_data, 
                                        train_y = train_y, 
                                        dev_data = dev_data, 
                                        dev_y = dev_y, 
                                        test_data = test_data, 
                                        test_y = test_y, 
                                        lr = learning_rate,
                                        epochs = epochs,
                                        split = split)
        
        
        torch.save(model.state_dict(), 
                   f"results/{self.run_name}/weights/nn_{split_num}_{symptom}.torch")
      
        ensemble_metrics, dev_output, test_output = self.ensemble(symptom, 
                                                                  description, 
                                                                  model, 
                                                                  train_data, 
                                                                  train_y, 
                                                                  dev_data, 
                                                                  dev_y, 
                                                                  test_data, 
                                                                  test_y, 
                                                                  split,
                                                                  split_num)  
        return ensemble_metrics, dev_output, test_output

        
    def train_model(self, 
                    symptom,
                    description,
                    model, 
                    train_data, 
                    train_y,
                    dev_data, 
                    dev_y,
                    test_data, 
                    test_y,
                    lr = 1e-4,
                    epochs = 400,
                    split = None):
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                      total_iters= epochs)     
        cw = compute_class_weight(class_weight = "balanced", classes = np.array([0,1]), y = dev_y.cpu().numpy())
        class_weight = torch.tensor(cw).cuda().float() if self.use_cuda else torch.tensor(cw).float()
        
        dev_metrics = self.evaluate(symptom, description, model, train_data, train_y, dev_data, dev_y, set_name = "dev")
        best_state = State(model, dev_metrics)  
        qbar = tqdm(range(epochs))
        for e in qbar:
            qbar.desc = self.training_step(model, train_data, train_y, optimizer, scheduler, class_weight=class_weight)
            dev_metrics = self.evaluate(symptom, description, model, train_data, train_y, dev_data, dev_y)
            self.results.update(dev_metrics, e, symptom)
            best_state.update(model, dev_metrics)
        model = best_state.restore(model)
        return best_state.dev_metrics
    
    def ensemble(self, 
                 symptom,
                 description,
                 model, 
                 train_data, 
                 train_y, 
                 dev_data, 
                 dev_y, 
                 test_data, 
                 test_y,
                 split,
                 split_num):
        train_X = torch.cat([train_data[f'genes:{f}'].x for f in train_data['genes'].features],1).cpu().numpy()
        train_y_np = train_y.cpu().numpy()          
        clf = LogisticRegression(class_weight="balanced")
        clf.fit(train_X,train_y_np)
        test_X = torch.cat([test_data[f'genes:{f}'].x for f in test_data['genes'].features],1).cpu().numpy()
        test_y_np = test_y.cpu().numpy()
        dev_X = torch.cat([dev_data[f'genes:{f}'].x for f in dev_data['genes'].features],1).cpu().numpy()
        dev_y_np = dev_y.cpu().numpy()
            
        clf_pred_test_y = clf.predict_proba(test_X)
        clf_pred_dev_y = clf.predict_proba(dev_X)
        with torch.no_grad():
            nn_pred_test_y = model(test_data).softmax(-1).cpu().numpy()
            nn_pred_dev_y = model(dev_data).softmax(-1).cpu().numpy()
        ens = (nn_pred_test_y + clf_pred_test_y)/2
        met_test_nn = get_row(symptom,
                              test_y_np[train_data['genes'].num_nodes:], 
                              nn_pred_test_y[train_data['genes'].num_nodes:].argmax(-1), 
                              set_name="test:nn")
        met_test_lr = get_row(symptom,
                              test_y_np[train_data['genes'].num_nodes:], 
                              clf_pred_test_y[train_data['genes'].num_nodes:].argmax(-1), 
                              set_name="test:lr")
        mets = {'symptom':symptom,
                'name':description}
        mets.update(met_test_nn)
        mets.update(met_test_lr)
        
        try:
            met_dev_nn = get_row(symptom,
                                 dev_y_np[train_data['genes'].num_nodes:], 
                                 nn_pred_dev_y[train_data['genes'].num_nodes:].argmax(-1), 
                                 set_name="dev:nn")
            met_dev_lr = get_row(symptom,
                                 dev_y_np[train_data['genes'].num_nodes:], 
                                 clf_pred_dev_y[train_data['genes'].num_nodes:].argmax(-1), 
                                 set_name="dev:lr")
            mets.update(met_dev_nn)
            mets.update(met_dev_lr)    
        except:
            pass
        met_ensemble = get_row(symptom,test_y_np[train_data['genes'].num_nodes:], ens[train_data['genes'].num_nodes:].argmax(-1), set_name="test")
        
        mets.update(met_ensemble)
    
        df_test_nn = pd.DataFrame(nn_pred_test_y[train_data['genes'].num_nodes:], index = split['test'], columns=[f"nn:test:{symptom}_{i}" for i in range(2)])
        df_test_lr = pd.DataFrame(clf_pred_test_y[train_data['genes'].num_nodes:], index = split['test'], columns=[f"lr:test:{symptom}_{i}" for i in range(2)])
        
        df_dev_nn = pd.DataFrame(nn_pred_dev_y[train_data['genes'].num_nodes:], index = split['dev'], columns=[f"nn:dev:{symptom}_{i}" for i in range(2)])
        df_dev_lr = pd.DataFrame(clf_pred_dev_y[train_data['genes'].num_nodes:], index = split['dev'], columns=[f"lr:dev:{symptom}_{i}" for i in range(2)])
        
        joblib.dump(clf,
                    f"results/{self.run_name}/linear/lr_{split_num}_{symptom}.torch")
        
        return mets, pd.merge(df_dev_nn,df_dev_lr, left_index=True,right_index=True), pd.merge(df_test_nn,df_test_lr, left_index=True,right_index=True)
        
    def training_step(self, 
                      model : torch.nn.Module, 
                      train_data : HeteroData, 
                      train_y : torch.Tensor, 
                      optimizer : torch.optim.Optimizer, 
                      scheduler : torch.optim.lr_scheduler.LRScheduler, 
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
        

        model.zero_grad(set_to_none=True)
        for v in train_data.x_dict.values():
            v.grad=None
        model.zero_grad()
        
        # Forward pass: Get model predictions
        pred_y = model.forward(train_data)
        
        # Calculate cross-entropy loss, applying class weights if provided
        loss = torch.nn.functional.cross_entropy(pred_y, train_y.long(), weight = class_weight)

        # Backward pass: Calculate gradients
        loss.backward(retain_graph=True)
        
        # Update model parameters based on gradients
        optimizer.step()
        
        # Update learning rate according to scheduler
        scheduler.step()
        return str(float(loss))
        
    def evaluate(self, symptom, description, model, train_data, train_y, dev_data, dev_y, set_name = "dev"):
        # import pdb
        # pdb.set_trace()
        metrics = {"symptom":symptom}
        with torch.no_grad():
            dev_pred = model(dev_data)
        train_pred_targets = [int(x) for x in (dev_pred[:len(train_y)].argmax(-1)).cpu().numpy()]
        train_true_targets = [int(x) for x in train_y]
        
        dev_pred_targets = [int(x) for x in (dev_pred[len(train_y):].argmax(-1)).cpu().numpy()]
        dev_true_targets = [int(x) for x in dev_y[len(train_y):]]
        
        metrics.update(get_row("", train_true_targets, train_pred_targets, set_name = "train"))
        try:
            metrics.update(get_row("", dev_true_targets, dev_pred_targets, set_name = set_name))
        except:
            pass
        return metrics