import numpy as np
from sklearn.metrics import (balanced_accuracy_score,
                             f1_score,
                             cohen_kappa_score,
                             matthews_corrcoef, 
                             roc_auc_score,
                             precision_recall_curve)
import torch
import pandas as pd

map_hpo_to_name =  pd.read_csv("data/raw/phenotype_to_genes.txt",sep="\t")
map_hpo_to_name = {hpo_id:name for hpo_id,name in zip(map_hpo_to_name['hpo_id'],map_hpo_to_name['hpo_name'])}


def get_fmax(preds: np.ndarray, 
             ys: np.ndarray, 
             beta = 1.0, 
             pos_label = 1):
    """
    Radivojac, P. et al. (2013). A Large-Scale Evaluation of Computational Protein Function Prediction. Nature Methods, 10(3), 221-227.
    """
    # import pdb
    # pdb.set_trace()
    precision, recall, thresholds = precision_recall_curve(y_true = ys, y_score = preds, pos_label = pos_label)
    numerator = (1 + beta**2) * (precision * recall)
    denominator = ((beta**2 * precision) + recall)
    fbeta = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator!=0))
    
    return np.nanmax(fbeta), thresholds[np.argmax(fbeta)]


def get_row(hpoid, y_true, y_pred, set_name = None):
    if set_name is not None:
        set_name = "_" + set_name
    row = {"hpoid":hpoid}
    row['name'] = map_hpo_to_name[hpoid] if hpoid in map_hpo_to_name else hpoid
    row[f'N{set_name}'] = int(sum(y_true))
    row[f'f1{set_name}'] = f1_score(y_true, y_pred)
    row[f'kappa{set_name}'] = cohen_kappa_score(y_true, y_pred)
    row[f'bacc{set_name}'] = balanced_accuracy_score(y_true, y_pred)
    row[f'matthews{set_name}'] = matthews_corrcoef(y_true, y_pred)
    row[f'Fmax{set_name}'] = get_fmax(y_pred, y_true)[0]
    try:
        row[f'auc_roc{set_name}'] = roc_auc_score(y_true, y_pred)
    except:
        row[f'auc_roc{set_name}'] = None
    return row

def compute_mcc_matrix(df: pd.DataFrame) -> pd.DataFrame:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(df.values, dtype=torch.float32, device=device)
    mcc_matrix = torch.corrcoef(X.T)
    mcc_matrix = torch.nan_to_num(mcc_matrix, nan=0.0)
    return pd.DataFrame(mcc_matrix.cpu().numpy(), index=df.columns, columns=df.columns)