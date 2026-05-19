import warnings 
import pandas as pd
from data import get_target_data
from graph.network_data import get_geometric_data
from model.core import Run
from utils.data_utils import set_index, print_summary
warnings.filterwarnings("ignore")

target_data = get_target_data(threshold = 100)
target_data = target_data.astype(int)

genes = list(target_data.index)

hidden = 512

data = get_geometric_data(target_data, genes=genes, split_number=0, selection_threshhold=10, hidden = hidden)

print_summary(data)

run = Run(data,
          target_data,
          hidden = hidden,
          dropout = 0.0,
          learning_rate=1e-3,
          epochs = 100, 
          run_name = "final")

run.run_model(n_trials = 1)
# df = run.results.report().to_csv("reults/initial_results.csv")
