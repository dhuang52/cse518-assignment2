import pandas as pd
import numpy as np
from scipy.special import comb

def expectation(D, l):
    """Return worker quality vector"""
    def _update_quality_score(s):
        # binom cdf
        N = l.size
        k = (s == l).sum()
        p = k / N
        c = comb(N, k)
        return c * (p**k) * ((1-p)**(N-k))
    return D.apply(_update_quality_score)

def maximization(D, theta):
    def _update_labels(s):
        nan_mask = ~pd.isna(s)
        s = s[nan_mask].values
        task_thetas = theta[nan_mask]
        return 1 if np.nansum(task_thetas*s) > 0 else -1
    return D.apply(_update_labels, axis=1)

def initialize_D(rte_data, w):
    def _get_responses_per_task(df):
        relevant_records = df.merge(w, how='right', on='!amt_worker_ids')
        return pd.Series(data=relevant_records['response'].values,
                        index=relevant_records['!amt_worker_ids'])
    return rte_data.groupby(['orig_id']).apply(_get_responses_per_task).unstack()
