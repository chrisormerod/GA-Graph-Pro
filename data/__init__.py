from .download_raw import (check_raw_data,
                           full_go_raw_path,
                           gene_embedding_raw_path)
from .parse_raw import (check_parsed_data,)
from .get_data import (get_target_data, get_descriptions)

def check_status():
    check_raw_data()
    check_parsed_data()
    
hpo_descriptions = get_descriptions()