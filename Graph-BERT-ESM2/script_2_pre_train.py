""" 
There are four changes in this file that are not present in the original `script_2_pre_train` in Graph-BERT folder. 

1. Reads dataset_name and embedder from PPI_DATASET / PPI_EMBEDDER env vars (set by run.sh), with fallback to in-file defaults.

2. Auto-detects nfeature and ngraph from the staged node file.

3. Tags result filenames with embedder so runs don't overwrite each other.

4. GPU selection + CPU core cap via env vars (handled by run.sh).

"""

import os
import numpy as np
import torch
 
from code.DatasetLoader import DatasetLoader
from code.MethodBertComp import GraphBertConfig
from code.MethodGraphBertNodeConstruct import MethodGraphBertNodeConstruct
from code.MethodGraphBertGraphRecovery import MethodGraphBertGraphRecovery
from code.ResultSaving import ResultSaving
from code.Settings import Settings
 
# ---- Seeding ----
np.random.seed(1)
torch.manual_seed(1)
 
# ---- Device selection ----
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed_all(1)
    print(
        f"Using GPU: {torch.cuda.get_device_name(0)} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')})"
    )
else:
    device = torch.device("cpu")
    print("Using CPU")
 
# ---- Dataset + embedder from env vars (set by run.sh) ----
dataset_name = os.environ.get('PPI_DATASET', 'c.elegan')
embedder     = os.environ.get('PPI_EMBEDDER', 'seqvec')
 
print(f"Dataset : {dataset_name}")
print(f"Embedder: {embedder}")
 
# ---- Auto-detect nfeature and ngraph from the staged node file ----
node_path = f'./data/{dataset_name}/node'
if not os.path.exists(node_path):
    raise FileNotFoundError(
        f"Staged node file not found: {node_path}\n"
        f"Run:  python stage_node.py --dataset {dataset_name} --embedder {embedder}\n"
        f"  or: ./run.sh <cores> <gpu> script_2_pre_train.py {dataset_name} {embedder}"
    )
 
with open(node_path) as f:
    first_line = f.readline().strip()
    ncols = len(first_line.split('\t'))
    nfeature = ncols - 2          # first col = pair_id, last col = label, rest = features
    ngraph = 1 + sum(1 for _ in f)  # 1 for the line already read + remaining
 
nclass = 2  # binary PPI classification for all datasets
 
print(f"Node file: {node_path}")
print(f"  ngraph   = {ngraph}")
print(f"  nfeature = {nfeature}")
print(f"  nclass   = {nclass}")
 
 
# ---- Pre-Training Task #1: Graph Bert Node Attribute Reconstruction ----
if 1:
    # ---- hyper-parameters ----
    lr = 0.001
    k = 7
    max_epoch = 200
 
    x_size = nfeature
    hidden_size = intermediate_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2
    y_size = nclass
    graph_size = ngraph
    residual_type = 'graph_raw'
 
    print('************ Start ************')
    print(f'GraphBert | dataset: {dataset_name} | embedder: {embedder} | '
          f'Pre-training: Node Attribute Reconstruction')
 
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = f'./data/{dataset_name}/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True
 
    bert_config = GraphBertConfig(
        residual_type=residual_type, k=k, x_size=nfeature, y_size=y_size,
        hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers,
    )
 
    method_obj = MethodGraphBertNodeConstruct(bert_config)
    method_obj = method_obj.to(device)
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr
 
    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/GraphBert/'
    result_obj.result_destination_file_name = (
        f'{dataset_name}_{embedder}_{k}_node_reconstruction'
    )
 
    setting_obj = Settings()
    evaluate_obj = None
 
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
 
    print('************ Finish ************')
 
 
# ---- Pre-Training Task #2: Graph Bert Network Structure Recovery ----
if 0:
    lr = 0.001
    k = 7
    max_epoch = 200
 
    x_size = nfeature
    hidden_size = intermediate_size = 32
    num_attention_heads = 2
    num_hidden_layers = 2
    y_size = nclass
    graph_size = ngraph
    residual_type = 'graph_raw'
 
    print('************ Start ************')
    print(f'GraphBert | dataset: {dataset_name} | embedder: {embedder} | '
          f'Pre-training: Graph Structure Recovery')
 
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = f'./data/{dataset_name}/'
    data_obj.dataset_name = dataset_name
    data_obj.k = k
    data_obj.load_all_tag = True
 
    bert_config = GraphBertConfig(
        residual_type=residual_type, k=k, x_size=nfeature, y_size=y_size,
        hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers,
    )
 
    method_obj = MethodGraphBertGraphRecovery(bert_config)
    method_obj.device = device
    method_obj = method_obj.to(device)
    method_obj.max_epoch = max_epoch
    method_obj.lr = lr
 
    result_obj = ResultSaving()
    result_obj.result_destination_folder_path = './result/GraphBert/'
    result_obj.result_destination_file_name = (
        f'{dataset_name}_{embedder}_{k}_graph_recovery'
    )
 
    setting_obj = Settings()
    evaluate_obj = None
 
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.load_run_save_evaluate()
 
    print('************ Finish ************')
 