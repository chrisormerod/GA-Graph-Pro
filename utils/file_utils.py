import gzip
import shutil
import h5py
import pandas as pd
import numpy as np
import zipfile

def unzip(gz_file : str, 
         filepath : str):
    try:
        # Opens a gzip-compressed file in binary read mode
        with gzip.open(gz_file, 'rb') as f_in:
            # Opens the target file in binary write mode
            with open(filepath, 'wb') as f_out:
                # Copies the content from the compressed file to the target file
                # shutil.copyfileobj efficiently copies file objects without loading
                # the entire content into memory at once
                shutil.copyfileobj(f_in, f_out)
    except:
        with zipfile.ZipFile(gz_file, 'r') as zip_ref:
            zip_ref.extractall(filepath)

def read_h5(filename: str) -> pd.DataFrame:
    """
    Reads a HDF5 file and returns a Pandas DataFrame containing gene-embedding pairs.

    Args:
        filename (str): The path to the HDF5 file.

    Returns:
        pd.DataFrame: A Pandas DataFrame where each row represents a gene and its corresponding embedding.
    """

    # Open the HDF5 file in read-only mode
    with h5py.File(filename, "r") as fp:

        # Initialize an empty dictionary to store the embeddings for each gene
        embedding = {}

        # Iterate over all groups (i.e., genes) in the HDF5 file
        for gene, vector in fp.items():
            # Convert the group's data into a NumPy array and add it to the 'embedding' dictionary
            embedding[gene] = np.array(vector)

    # Create a Pandas DataFrame from the 'embedding' dictionary
    # The transpose operation is used because HDF5 stores embeddings as vectors, not matrices
    embedding_df = pd.DataFrame(embedding).transpose()

    # Return the resulting DataFrame
    return embedding_df
