import pandas as pd
import numpy as np
import csv

def true_labels(rte_data):
    """ Return Series, index orig_id, gold of task
    rte_data (DataFrame)
    """
    return rte_data.groupby(['orig_id']).first()['gold']

def get_k_tasks(rte_data, k, seed=518):
    """ Return DataFrame where k annotations are randomly drawn for each sentence pair
    rte_data (DataFrame)
    k (int): k = 1, 2, ..., 10 (randomly chosen)
    """
    return rte_data.groupby(['orig_id'], as_index=False)\
                        .apply(lambda df: df.sample(k))

def get_k_workers(worker_ids, k, seed=518):
    """ Return set of k worker id's
    worker_ids (nparray)
    k (int): k = 1, 2, ..., 10 (randomly chosen)
    """
    return pd.Series(np.random.choice(worker_ids, k), name='!amt_worker_ids')

def agg_error(predictions, true_task_labels):
    """ Return error rate
    predictions (Series)
    true_task_labels (Series)
    """
    correct = (predictions == true_task_labels).sum()
    return 1 - (correct / predictions.size)

def write_to_csv(file_name, avg_agg_errors):
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in avg_agg_errors.items():
           writer.writerow([key, value])
