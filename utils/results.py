#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 20:58:33 2025

@author: cormerod
"""
import os
import json
import pandas as pd
from typing import Dict

class ResultLog:
    
    def __init__(self, 
                 target_data : pd.DataFrame,
                 config: Dict,
                 logging_epochs : int,
                 name : str,
                 split_num : int):
        print(f"Creating {name} result log")
        self.training_log = {s:[] for s in target_data.columns}
        self.log = {}
        self.logging_epochs = logging_epochs
        self.name = name
        self.split_num = split_num
        self.log_filename = f"results/{self.name}/log-{self.split_num}.json"
        self.read_directory()
        self.save()
    
            
    def read_directory(self):
        if os.path.exists(self.log_filename):
            with open(self.log_filename, "r") as fp:
                self.log = json.load(fp)
        if len(self.log) > 0:
            print(f"Read {len(self.log)} previous symptoms" )
            
                
    def save(self):
        with open(self.log_filename, "w") as fp:
            json.dump(self.log, fp, indent = 2)
        
    def check_complete(self, symptom):
        return True if symptom in self.log else False
    
    def update(self, 
               metrics : Dict,
               epoch : int,
               s: str):
        if epoch % self.logging_epochs == self.logging_epochs - 1:
            self.training_log[s].append(metrics)
    
    def finalize(self, symptom, metrics):
        self.log[symptom] = metrics
        self.save()
        return self.get_desc()
        
    def get_desc(self):
        df = self.report()
        f1 = sum(df['f1_test'] >= 0.2)
        kappa = sum(df['kappa_test'] >= 0.2)
        bacc = sum(df['bacc_test'] >= 0.6)
        return f"f1:{f1} k:{kappa} bacc:{bacc}"
        
    def report(self):
        return pd.DataFrame(self.log).transpose()