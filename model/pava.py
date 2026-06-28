# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:40:57 2026

@author: chris
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- Data Structures as described in the problem ---

class SymptomNode:
    """A simple node to represent a symptom in the hierarchy."""
    def __init__(self, name, hpo_id=None):
        self.name = name
        self.hpo_id = hpo_id
        self.children = {}

    def add_child(self, child_node):
        self.children[child_node.name] = child_node

    def __repr__(self):
        return f"SymptomNode({self.name}, {self.hpo_id})"

# --- OBO Parser ---

def parse_hpo_file(filepath):
    """
    Parses an .obo file to build a dictionary of SymptomNode objects.
    Returns:
        nodes: Dict mapping HPO IDs to SymptomNode objects.
        hierarchy: Dict where keys are root nodes (nodes with no parents).
    """
    nodes = {}  # hpo_id -> SymptomNode
    child_parent_map = [] # list of (child_id, parent_id)
    
    print(f"Parsing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        current_term = None
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_term = {}
            elif line == "" and current_term:
                # End of term block: create the node
                if 'id' in current_term and 'name' in current_term:
                    nodes[current_term['id']] = SymptomNode(current_term['name'], current_term['id'])
                current_term = None
            elif current_term is not None:
                if line.startswith("id: "):
                    current_term['id'] = line[4:].split('!')[0].strip()
                elif line.startswith("name: "):
                    current_term['name'] = line[6:].strip()
                elif line.startswith("is_a: "):
                    parent_id = line[6:].split('!')[0].strip()
                    child_parent_map.append((current_term['id'], parent_id))

    # Link the nodes to build the actual tree
    print("Linking nodes into hierarchy...")
    child_ids = set()
    for child_id, parent_id in child_parent_map:
        if child_id in nodes and parent_id in nodes:
            nodes[parent_id].add_child(nodes[child_id])
            child_ids.add(child_id)

    # Roots are nodes that were never listed as a child
    roots = {node.name: node for id, node in nodes.items() if id not in child_ids}
    
    print(f"  - Total nodes: {len(nodes)}")
    print(f"  - Root symptoms found: {list(roots.keys())}")
    return nodes, roots

# --- Core PAVA Implementation (Adapted for HPO IDs) ---

def get_parent_child_pairs(symptoms_dict):
    pairs = []
    nodes_to_visit = list(symptoms_dict.values())
    visited = set()

    while nodes_to_visit:
        current_node = nodes_to_visit.pop(0)
        if current_node.hpo_id in visited:
            continue
        visited.add(current_node.hpo_id)
        
        for child_name, child_node in current_node.children.items():
            pairs.append((current_node.hpo_id, child_node.hpo_id))
            nodes_to_visit.append(child_node)
            
    return pairs

def apply_pava(merged_df: pd.DataFrame, symptoms_hierarchy: dict) -> pd.DataFrame:
    parent_child_pairs = get_parent_child_pairs(symptoms_hierarchy)
    
    col_to_idx = {col: i for i, col in enumerate(merged_df.columns)}
    
    active_pairs_idx = [
        (col_to_idx[parent_id], col_to_idx[child_id]) 
        for parent_id, child_id in parent_child_pairs 
        if parent_id in col_to_idx and child_id in col_to_idx
    ]

    if not active_pairs_idx:
        return merged_df.copy()

    data = merged_df.to_numpy(dtype=np.float64, copy=True)
    max_iterations = 10000
    
    for _ in tqdm(range(max_iterations), desc="PAVA Iterations"):
        total_violations_this_iter = 0

        for p_idx, c_idx in active_pairs_idx:
            violators_mask = data[:, p_idx] < data[:, c_idx]
            num_violators = violators_mask.sum()
            
            if num_violators > 0:
                total_violations_this_iter += num_violators
                
                avg_prob = (data[violators_mask, p_idx] + data[violators_mask, c_idx]) / 2.0 
                
                data[violators_mask, p_idx] = avg_prob
                data[violators_mask, c_idx] = avg_prob

        if total_violations_this_iter == 0:
            break
            
    return pd.DataFrame(data, index=merged_df.index, columns=merged_df.columns)

# --- Example of how to call it with your parser ---
# all_nodes, roots = parse_hpo_file("../data/raw/hp.obo")
# reconciled_results = apply_pava(merged_df, all_nodes)

# --- Example Usage ---

# if __name__ == '__main__':
    # 1. Define the Symptom Hierarchy
    # HP:0000118 -> Phenotypic abnormality
    #   -> HP:0001249 -> Intellectual disability
    #      -> HP:0002187 -> Global developmental delay
    #   -> HP:0000707 -> Abnormality of the nervous system
    #      -> HP:0001250 -> Seizure
    all_nodes, roots = parse_hpo_file("../data/raw/hp.obo")
    
    # root = SymptomNode("Phenotypic abnormality")
    # intel_disability = SymptomNode("Intellectual disability")
    # gdd = SymptomNode("Global developmental delay")
    # nervous_abnormality = SymptomNode("Abnormality of the nervous system")
    # seizure = SymptomNode("Seizure")

    # # Build the tree structure
    # root.add_child(intel_disability)
    # root.add_child(nervous_abnormality)
    # intel_disability.add_child(gdd)
    # nervous_abnormality.add_child(seizure)

    # # The main symptoms dictionary passed to the function
    # symptoms = {"Phenotypic abnormality": root}
    # symptom_names = [
    #     "Phenotypic abnormality", "Intellectual disability",
    #     "Global developmental delay", "Abnormality of the nervous system", "Seizure"
    # ]

    # # 2. Generate mock data for the example
    # create_mock_data(symptom_names, num_genes=5)

    # # 3. Run the PAVA reconciliation
    # reconciled_results = apply_pava(symptoms, symptom_names)

    # # 4. Display a sample of the results for verification
    # if reconciled_results:
    #     print("\n--- Sample Reconciled Probabilities for 'Global developmental delay' ---")
    #     gdd_df = reconciled_results.get("Global developmental delay")
    #     if gdd_df is not None:
    #         print(gdd_df.head())
        
    #     print("\n--- Sample Reconciled Probabilities for its parent 'Intellectual disability' ---")
    #     id_df = reconciled_results.get("Intellectual disability")
    #     if id_df is not None:
    #         print(id_df.head())

    #     # Verification: Check if P(parent) >= P(child) holds
    #     if gdd_df is not None and id_df is not None:
    #         comparison = (id_df['p_1'] >= gdd_df['p_1']).all()
    #         print(f"\nVerification Check: P(Intellectual disability) >= P(Global developmental delay) for all genes? -> {comparison}")

    # # 5. Clean up the generated files
    # print("\nCleaning up mock data files...")
    # cleanup_mock_data(symptom_names)