from .download_raw import (full_go_raw_path,
                           pathway_raw_path,
                           complex_raw_path,
                           gene_embedding_raw_path,
                           mapping_uniprot_symbol_raw_path,
                           mouse_expression_raw_path,
                           subcellular_location_raw,
                           normalize_tissue_raw,
                           curated_correspondences_file)
import pandas as pd
import numpy as np
from utils.data_utils import one_hot_dataframe
from utils.file_utils import read_h5
import re

mapping_uniprot_name_file = "data/generated/correspondences/mapping_uniprot_name.json"
full_go_file = "data/generated/gene/full_go_terms.feather"
pathway_file = "data/generated/gene/pathways.feather"
complex_file = "data/generated/gene/complex.feather"
embedding_file = "data/generated/gene/embedding.feather"
subcellular_location_file = "data/generated/gene/subcellular_location.feather"

# target_data_file = "data/generated/targets/gold_standard.feather"
target_data_file = "data/generated/gemini.feather"

def check_parsed_data():
    pass

def parse_embedding():
    """
    Reads an HDF5 file containing gene-embedding pairs and saves them to a Feather file.

    This function assumes that it has been called previously with the result of `read_h5` (i.e., 
    `gene_embedding_raw_path`) as its argument.
    """

    # Call the read_h5 function to load the embedding data from the HDF5 file
    embedding_df = read_h5(gene_embedding_raw_path)
    # Save the DataFrame containing the embeddings to a Feather file
    embedding_df.to_feather(embedding_file)

def parse_complexes():
    # Read raw complex data file line by line
    with open(complex_raw_path, "r") as fp:
        lines = fp.readlines()
    
    # Create DataFrame from tab-separated data, using first line as column headers
    data = pd.DataFrame(data = [x.split("\t") for x in lines[1:]], 
                        columns = lines[0].split("\t"))
    
    # Extract UniProt IDs from the participants column
    # For each row, split on |, keep only entries starting with "UniProt", and remove "UniProt" prefix
    data['uniprot_ids'] = data['participants'].map(lambda x:[y[8:] for y in x.split("|") if y.startswith("UniProt")])
    
    # Get unique values for rows (flattened list of all UniProt IDs) and columns (complex identifiers)
    indices = list(set(sum(data['uniprot_ids'], list())))  # Flatten list of lists and get unique values
    columns = list(set(data['identifier']))
    
    # Create mapping dictionaries to convert labels to numeric indices for faster array operations
    idx_map = {k:v for v,k in enumerate(indices)}  # Maps UniProt IDs to integers
    col_map = {k:v for v,k in enumerate(columns)}  # Maps complex identifiers to integers
    
    # Initialize empty binary matrix with dimensions: (number of unique proteins) x (number of unique complexes)
    A = np.zeros(shape = (len(indices), len(columns)))
    
    # Fill matrix: set 1 if protein (row) is part of complex (column)
    for L, col in zip(data['uniprot_ids'], data['identifier']):
        for i in L:  # For each protein in the complex
            A[idx_map[i], col_map[col]] = 1
    
    # Convert numpy array to DataFrame with proper labels
    feature = pd.DataFrame(A, index = indices, 
                           columns = columns)
    
    # Save result to feather file format (efficient binary format for DataFrames)
    feature.to_feather(complex_file)    

def parse_full_go_terms():
    # Open and read the GO terms file
    with open(full_go_raw_path,"r") as fp:
        lines = fp.readlines()
    
    # Filter out comment lines (starting with !) and split each line by tabs
    # This creates a list of lists where each inner list contains tab-separated values
    lines = [x.split("\t") for x in lines if not x.startswith("!")]
    
    # Convert the filtered lines to a pandas DataFrame
    # Select only columns at index 2 and 4 (likely containing GO terms and their annotations)
    data = pd.DataFrame(lines)[[2,4]]
    
    # Create one-hot encoded features from the GO terms data
    # Converts categorical variables in columns 2 and 4 into binary columns
    feature = one_hot_dataframe(data[2], data[4])    
    
    # Save the one-hot encoded features to a feather file format
    # Feather is an efficient file format for storing DataFrame objects
    feature.to_feather(full_go_file)

def parse_pathway_terms():
    # Open and read the pathway terms file
    with open(pathway_raw_path) as fp:
        lines = fp.readlines()
    
    # Split each line by tabs to create a list of lists
    # Each inner list contains the tab-separated values from one line
    lines = [x.split("\t") for x in lines]
    
    # Convert the list of lists into a pandas DataFrame
    data = pd.DataFrame(lines)
    
    # Filter the DataFrame to only keep rows where column 5 contains "Homo sapiens"
    # This restricts the data to human-specific pathways
    data = data[data[5] == "Homo sapiens\n"]
    
    # Create one-hot encoded features from columns 0 and 1
    # Column 0 likely contains pathway IDs and column 1 contains pathway annotations
    feature = one_hot_dataframe(data[0], data[1])
    
    # Save the one-hot encoded features to a feather file format
    # Feather is an efficient file format for storing DataFrame objects
    feature.to_feather(pathway_file)


def parse_subcellular_location():
    # Load raw subcellular location data from a tab-separated file
    data = pd.read_csv(subcellular_location_raw, sep="\t")
    
    # Initialize empty lists to store gene names and their corresponding subcellular locations
    genes = []
    locations = []
    
    # Create a new column 'location_list' that splits the 'Main location' column by semicolons
    # This converts a string like "nucleus;cytoplasm" into a list ["nucleus", "cytoplasm"]
    data['location_list'] = data['Main location'].map(lambda x: str(x).split(";"))
    
    # Iterate through each gene and its corresponding list of locations
    for name, gene_locations in zip(data['Gene name'], data['location_list']):
        # For each location in the gene's location list
        for loc in gene_locations:
            # Only process non-empty locations (skip 'nan' values)
            if loc != "nan":
                # Add the gene name and location as a pair to their respective lists
                genes.append(name)
                locations.append(loc)
    
    # Convert the paired gene-location data into a one-hot encoded dataframe
    # Each row corresponds to a gene, and columns represent different locations
    # Values are binary (1 if the gene is found in that location, 0 otherwise)
    df = one_hot_dataframe(pd.Series(genes), pd.Series(locations))
    
    # Save the processed dataframe to a feather file for efficient storage and future use
    df.to_feather(subcellular_location_file)




def parse_mapping_uniport():
    """
    Reads a UniProt mapping file, parses it to create a dictionary 
    where each key is an Uniprot ID and its corresponding value is a list of Gene Names.
    
    Returns:
        None. The parsed data is written to the `mapping_uniprot_name_file` path.
    
    Raises:
        FileNotFoundError: If the input file at `mapping_uniprot_symbol_raw_path` does not exist.
    """
    
    # Open the raw UniProt mapping file in read mode
    with open(mapping_uniprot_symbol_raw_path, "r") as fp:
        
        # Read all lines from the file and store them in a list
        lines = fp.readlines()
    
    # Filter out non-Gene_Name lines (e.g., header or empty lines)
    lines = [l.split("\t") for l in lines if "Gene_Name" in l]
    
    # Initialize an empty dictionary to store UniProt IDs as keys and Gene Names as values
    mapping_uniprot_name = {}
    
    # Iterate over each line in the filtered list of lines
    for uniprot, _, gene in lines:
        
        # If a Uniprot ID is already present in the dictionary, append its corresponding Gene Name to the existing list
        if uniprot in mapping_uniprot_name:
            # Strip any leading/trailing whitespace from the Gene Name before appending it
            mapping_uniprot_name[uniprot].append(gene.strip())
            
            # Print the current UniProt ID for debugging purposes (optional)
            print(uniprot)
        
        # If a Uniprot ID is not present in the dictionary, add it with its corresponding Gene Name as a list
        else:
            # Strip any leading/trailing whitespace from the Gene Name before adding it to the dictionary
            mapping_uniprot_name[uniprot] = [gene.strip()]
    
    # Open the output file path to write the parsed data
    with open(mapping_uniprot_name_file, "w") as fp:
        
        # Use json.dump() to serialize and dump the dictionary into a JSON-formatted string
        import json  # Import the json module (not shown in this code snippet)
        json.dump(mapping_uniprot_name,
                  fp,  # Specify the file object for writing
                  indent=2)  # Pretty-print the output with indentation   

    