import pandas as pd
import numpy as np
from scipy.linalg import svd

def find_top_worker(D, rte_gold):
    """
    return (str) worker id
    """
    def _apply(s):
        nan_mask = ~pd.isna(s)
        return s == rte_gold.loc[s.index]
    num_correct = D.apply(_apply).sum()
    return num_correct.idxmax()

def run_svd(D, w_1):
    D.fillna(0, inplace=True)
    D = D.to_numpy()
    U, S, V = svd(D@D.T)
    # top eigenvector
    top_eigen_v = U[0,:]
    sign_v = np.sign(top_eigen_v)
    for i in range(sign_v.size):
        w_1.iloc[i] =  w_1.iloc[i] if sign_v[i] == w_1.iloc[i] else -w_1.iloc[i]
    return w_1
