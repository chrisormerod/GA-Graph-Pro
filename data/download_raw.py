import urllib
from utils import unzip
import os

raw_dir = "data/raw"

gene_embedding_url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/embeddings/UP000005640_9606/per-protein.h5"
gene_embedding_raw_path = "data/raw/per-protein.h5"

full_go_url = "https://current.geneontology.org/annotations/goa_human.gaf.gz"
full_go_raw_path_gz = "data/raw/goa_human.gaf.gz"
full_go_raw_path = "data/raw/goa_human.gaf"

mapping_uniprot_symbol_url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz"
mapping_uniprot_symbol_raw_path_gz = "data/raw/HUMAN_9606_idmapping.dat.gz"
mapping_uniprot_symbol_raw_path = "data/raw/HUMAN_9606_idmapping.dat"

mouse_expression_url = "https://fms.alliancegenome.org/download/EXPRESSION-ALLIANCE_MGI.tsv.gz"
mouse_expression_raw_path_gz = "data/raw/EXPRESSION-ALLIANCE_MGI.tsv.gz"
mouse_expression_raw_path = "data/raw/EXPRESSION-ALLIANCE_MGI.tsv"

pathway_url = "https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt"
pathway_raw_path = "data/raw/UniProt2Reactome_All_Levels.txt"

complex_url = "https://reactome.org/download/current/ComplexParticipantsPubMedIdentifiers_human.txt"
complex_raw_path = "data/raw/ComplexParticipantsPubMedIdentifiers_human.txt"

hpo_project_target_url = "https://purl.obolibrary.org/obo/hp/phenotype_to_genes.txt"
hpo_project_target_raw = "data/raw/phenotype_to_genes.txt"

subcellular_location_url = "https://www.proteinatlas.org/download/tsv/subcellular_location.tsv.zip"
human_atlas_levels_url = "https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip"
normalize_tissue_url = "https://www.proteinatlas.org/download/tsv/normal_ihc_data.tsv.zip"
subcellular_location_raw_zip = "data/raw/subcellular_location.tsv.zip"
human_atlas_levels_raw_zip = "data/raw/rna_tissue_consensus.tsv.zip"
normalize_tissue_raw_zip = "data/raw/normal_ihc_data.tsv.zip"
subcellular_location_raw = "data/raw/subcellular_location.tsv"
human_atlas_levels_raw = "data/raw/rna_tissue_consensus.tsv"
normalize_tissue_raw = "data/raw/normal_ihc_data.tsv"

biogrid_url = "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.242/BIOGRID-ALL-4.4.242.mitab.zip"
biogrid_raw_zip = "data/raw/BIOGRID-ALL-4.4.242.mitab.zip"
biogrid_raw = "data/raw/BIOGRID-ALL-4.4.242.mitab.txt"

curated_correspondences_file = "data/raw/curated_correspondences.xlsx"
curated_correspondences_tabula_file = "data/generated/correspondences/hpo_tissue_correspondence.xlsx"

expression_alliance_url = "https://fms.alliancegenome.org/download/EXPRESSION-ALLIANCE_MGI.tsv.gz"
expression_alliance_raw_gz = "data/raw/EXPRESSION-ALLIANCE_MGI.tsv.gz"
expression_alliance_raw = "data/raw/EXPRESSION-ALLIANCE_MGI.tsv"

mouse_mapping_url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz"
mouse_mapping_raw_gz = "data/raw/Mus_musculus.gene_info.gz"
mouse_mapping_raw = "data/raw/Mus_musculus.gene_info"

human_mapping_url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
human_mapping_raw_gz = "data/raw/Homo_sapiens.gene_info.gz" 
human_mapping_raw = "data/raw/Homo_sapiens.gene_info"

def check_raw_data():
    if not os.path.exists(mapping_uniprot_symbol_raw_path):
        get_uniprot_mapping()
    if not os.path.exists(full_go_raw_path):
        get_go_terms()
    if not os.path.exists(pathway_raw_path):
        get_pathways()
    if not os.path.exists(complex_raw_path):
        get_complexes()
    if not os.path.exists(gene_embedding_raw_path):
        get_gene_embedding()
    if not os.path.exists(mouse_expression_raw_path):
        get_gene_embedding()
    if not os.path.exists(hpo_project_target_raw):
        get_hpo_project_targets()
    if not os.path.exists(subcellular_location_raw) or not os.path.exists(human_atlas_levels_raw) or not  os.path.exists(normalize_tissue_raw):
        get_human_atlas()
    if not os.path.exists(expression_alliance_raw):
        get_expression_alliance()
            
def get_expression_alliance():
    __get_url(expression_alliance_url, expression_alliance_raw_gz, expression_alliance_raw)
    __get_url(mouse_mapping_url, mouse_mapping_raw_gz, mouse_mapping_raw)
    __get_url(human_mapping_url, human_mapping_raw_gz, human_mapping_raw)

def get_human_atlas():
    __get_url(subcellular_location_url, subcellular_location_raw_zip, raw_dir)
    __get_url(human_atlas_levels_url,human_atlas_levels_raw_zip, raw_dir)
    __get_url(normalize_tissue_url, normalize_tissue_raw_zip, raw_dir)

def get_biogrid():
    __get_url(biogrid_url, biogrid_raw_zip, raw_dir)

def __get_url(url, path, unzip_path = None):
    urllib.request.urlretrieve(url, path)
    if unzip_path is not None:
        unzip(path, unzip_path)

def get_hpo_project_targets():
    __get_url(hpo_project_target_url,
              hpo_project_target_raw)

def get_uniprot_mapping():
    __get_url(mapping_uniprot_symbol_url,
              mapping_uniprot_symbol_raw_path_gz,
              mapping_uniprot_symbol_raw_path)

def get_pathways():
    __get_url(pathway_url, 
              pathway_raw_path)

def get_go_terms():
    __get_url(full_go_url,
              full_go_raw_path_gz,
              full_go_raw_path)
    
def get_complexes():
    __get_url(complex_url,
              complex_raw_path)
    
def get_gene_embedding():
    __get_url(gene_embedding_url,
              gene_embedding_raw_path)
    
def get_mouse_expression():
    __get_url(mouse_expression_url,
              mouse_expression_raw_path_gz,
              mouse_expression_raw_path)