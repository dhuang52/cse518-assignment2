import pandas as pd
import numpy as np
from majority_vote import count_labels, mv
from expectation_maximization import initialize_D, expectation, maximization
from svd import find_top_worker, run_svd
from utility import true_labels, get_k_tasks, get_k_workers, agg_error, write_to_csv
import math
from collections import defaultdict

"""
- !amt_worker_ids: The IDs of workers
– orig_id: The IDs of sentence pairs
– response: The worker’s answer (annotation) to the sentence pair (0 or 1)
– gold: Ground truth of the sentence pair
"""

"""
TIE BREAKING STRATEGY:
    for 1: must be strictly greater than threshold
"""

def run_mv_experiment(rte_k, rte_gold):
    """ Print error rate of MV over n experiments randomly sampling k
        annotations per task
    rte_k (DataFrame): dataset (k annotations per task)
    rte_gold (Series): gold of tasks
    """
    avg_error_rate = 0
    label_counts = count_labels(rte_k)
    mv_predictions = mv(label_counts)
    return agg_error(mv_predictions, rte_gold)

def run_em_experiment(D, w, rte_gold, iter):
    """ Perform EM experiment
    D (DataFrame): index: orig_id, column: worker_id
    w (Series): worker ids
    rte_gold (Series): gold of tasks
    iter (int): max number of iterations
    """
    # Initial setup
    theta = pd.Series(np.ones(w.size), index=w)
    l = pd.Series(np.random.choice([1, -1], D.index.size), index=D.index)
    n = rolling_sq_diff = mu = sigma = 0
    while n != iter:
        prev_l, prev_theta = np.array(l), np.array(theta)
        l = maximization(D, theta)
        theta = expectation(D, l)
        err = agg_error(l, rte_gold)
        n += 1
        # calculate std
        mu = ((mu * (n-1))+err) / n
        rolling_sq_diff += (err - mu)**2
        sigma = math.sqrt(rolling_sq_diff / n)
        if (sigma < .01 and n != 1) or (np.array_equiv(prev_l, l) and np.array_equiv(prev_theta, prev_theta)):
            break
    return agg_error(l, rte_gold)

def run_svd_experiment(D, rte_gold):
    """ Perform SVD experiment
    D (DataFrame): index: orig_id, column: worker_id
    rte_gold (Series): gold of tasks
    """
    w_1_id = find_top_worker(D, rte_gold)
    w_1 = D[w_1_id]
    w_1 = run_svd(D, w_1)
    return agg_error(w_1, rte_gold)

def agg_error_k(n, rte_data, rte_gold):
    """ For k = 1 ... 10, calculate aggregation error of each algorithm over n
        experiments, return them and sav to csv
    n (int): number of experiments to run
    rte_data (DataFrame): entire dataset
    rte_gold (Series): gold of each task
    return (List): list of avg errors over k = 1 ... 10
    """
    avg_mv_agg_errors = defaultdict(int)
    avg_em_agg_errors = defaultdict(int)
    avg_svd_agg_errors = defaultdict(int)
    for k in range(1, 11):
        for i in range(n):
            rte_k = get_k_tasks(rte_data, k)
            w = pd.Series(rte_k['!amt_worker_ids'].unique(), name='!amt_worker_ids')
            D = initialize_D(rte_k, w)
            avg_mv_agg_errors[k] += run_mv_experiment(rte_k, rte_gold)
            avg_em_agg_errors[k] += run_em_experiment(D, w, rte_gold, 20)
            avg_svd_agg_errors[k] += run_svd_experiment(D, rte_gold)
        avg_mv_agg_errors[k] = round(avg_mv_agg_errors[k] / n, 5)
        avg_em_agg_errors[k] = round(avg_em_agg_errors[k] / n, 5)
        avg_svd_agg_errors[k] = round(avg_svd_agg_errors[k] / n, 5)
    write_to_csv('avg_mv_agg_error.csv', avg_mv_agg_errors)
    write_to_csv('avg_em_agg_error.csv', avg_em_agg_errors)
    write_to_csv('avg_svd_agg_error.csv', avg_svd_agg_errors)
    return [avg_mv_agg_errors, avg_em_agg_errors, avg_svd_agg_errors]

# General setup
file_name = 'rte.standardized.tsv'
rte_data = pd.read_csv(file_name, sep='\t')[['!amt_worker_ids', 'orig_id',
                                            'response', 'gold']]

# map 0 -> -1
rte_data.replace(0, -1, inplace=True)
rte_gold = true_labels(rte_data)
n = 100
results = agg_error_k(n, rte_data, rte_gold)
for r in results:
    print(r)
