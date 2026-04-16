"""
There are four changes in this file that are not present in the original `script_4_evaluation_plots` in Graph-BERT folder. 

1. Added --embedder CLI arg to select which embedder's results to evaluate.

2. Result filename lookup uses the embedder-tagged naming convention from the patched script_3_fine_tuning.py.

3. When --embedder is 'all', loops over all three and prints a comparison table.
""" 
import argparse
 
import numpy as np
import torch
import torch.nn.functional as F
 
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)
 
from code.ResultSaving import ResultSaving
 
 
DATASETS = ['ppi', 'human', 'e.coli', 'drosophila', 'c.elegan']
EMBEDDERS = ['seqvec', 'esm2_650M', 'esm2_3B']
HIDDEN_LAYERS = 2
RESULT_FOLDER = './result/GraphBert/'
 
METRIC_KEYS = ['accuracy', 'sensitivity', 'specificity', 'precision',
               'recall', 'f1', 'mcc', 'auroc', 'auprc']
 
 
def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)
 
 
def evaluate_epoch(record):
    """Compute all metrics for a single saved epoch record."""
    y_true = to_numpy(record['test_acc_data']['true_y'])
 
    test_op = record['test_op']
    if not torch.is_tensor(test_op):
        test_op = torch.tensor(test_op)
    probs = F.softmax(test_op, dim=1).detach().cpu().numpy()
 
    y_pred = probs.argmax(axis=1)
    y_score = probs[:, 1]
 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
 
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
 
    return {
        'accuracy':    accuracy_score(y_true, y_pred),
        'sensitivity': tp / (tp + fn) if (tp + fn) else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) else 0.0,
        'precision':   precision,
        'recall':      recall,
        'f1':          f1,
        'mcc':         matthews_corrcoef(y_true, y_pred),
        'auroc':       roc_auc_score(y_true, y_score),
        'auprc':       average_precision_score(y_true, y_score),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }
 
 
def run_dataset(dataset_name, embedder_name):
    """Evaluate a single (dataset, embedder) run. Returns metrics dict or None."""
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = RESULT_FOLDER
    result_obj.result_destination_file_name = (
        f'{dataset_name}_{embedder_name}_{HIDDEN_LAYERS}'
    )
 
    try:
        history = result_obj.load()
    except (FileNotFoundError, IOError):
        print(f'\n  [{dataset_name} / {embedder_name}] no saved history at '
              f'{RESULT_FOLDER}{result_obj.result_destination_file_name} — skipping')
        return None
 
    epochs = sorted(history.keys())
    if not epochs:
        print(f'\n  [{dataset_name} / {embedder_name}] history is empty — skipping')
        return None
 
    # Select best epoch by test accuracy (legacy) and by validation accuracy (corrected)
    best_test_epoch = max(epochs, key=lambda e: history[e]['acc_test'])
    best_val_epoch  = max(epochs, key=lambda e: history[e]['acc_val'])
 
    legacy    = evaluate_epoch(history[best_test_epoch])
    corrected = evaluate_epoch(history[best_val_epoch])
 
    same_epoch = (best_test_epoch == best_val_epoch)
    identical  = same_epoch and all(
        np.isclose(legacy[k], corrected[k])
        for k in legacy if isinstance(legacy[k], float)
    )
 
    print(f'\n  [{dataset_name} / {embedder_name}]  '
          f'epochs: {len(epochs)} ({epochs[0]}..{epochs[-1]})')
    print(f'    legacy (argmax acc_test) → epoch {best_test_epoch}')
    print(f'    corrected (argmax acc_val) → epoch {best_val_epoch}')
    if identical:
        print('    (legacy == corrected on this dataset; idx_val == idx_test)')
    print(f'    {"metric":<12s} {"legacy":>10s} {"corrected":>10s} {"delta":>10s}')
    for k in METRIC_KEYS:
        delta = corrected[k] - legacy[k]
        print(f'    {k:<12s} {legacy[k]:>10.4f} {corrected[k]:>10.4f} '
              f'{delta:>+10.4f}')
 
    return corrected  # return the corrected metrics for the comparison table
 
 
def run_comparison(dataset_name, embedder_list):
    """Run all embedders for one dataset and print a comparison table."""
    print(f'\n{"=" * 72}')
    print(f'  DATASET: {dataset_name}')
    print(f'{"=" * 72}')
 
    results = {}
    for emb in embedder_list:
        metrics = run_dataset(dataset_name, emb)
        if metrics is not None:
            results[emb] = metrics
 
    if len(results) < 2:
        return
 
    # Print side-by-side comparison table
    print(f'\n  {"─" * 60}')
    print(f'  Comparison (corrected / best-val-epoch):')
    header = f'  {"metric":<12s}'
    for emb in results:
        header += f' {emb:>12s}'
    print(header)
 
    for k in METRIC_KEYS:
        row = f'  {k:<12s}'
        for emb in results:
            row += f' {results[emb][k]:>12.4f}'
        print(row)
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset', default='all',
        help=f'Dataset name, or "all" to loop over {DATASETS}. Default: all.',
    )
    parser.add_argument(
        '--embedder', default='all',
        choices=EMBEDDERS + ['all'],
        help=f'Embedder variant, or "all" to compare. Default: all.',
    )
    args = parser.parse_args()
 
    datasets  = DATASETS if args.dataset == 'all' else [args.dataset]
    embedders = EMBEDDERS if args.embedder == 'all' else [args.embedder]
 
    for d in datasets:
        if len(embedders) > 1:
            run_comparison(d, embedders)
        else:
            print(f'\n{"=" * 72}')
            print(f'  DATASET: {d}')
            print(f'{"=" * 72}')
            run_dataset(d, embedders[0])
 
 
if __name__ == '__main__':
    main()