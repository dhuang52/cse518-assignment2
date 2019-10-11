import pandas as pd

def count_labels(rte_data):
    """ Return Series, index orig_id, count of '1' votes
    rte_data (DataFrame)
    """
    return rte_data.groupby(['orig_id'])['response'].apply(lambda s: s[s > 0].sum() / s.size)

def mv(label_counts):
    """ Return Series, index orig_id, crowds prediction
    task_label_counts (Series)
    """
    return label_counts.apply(lambda n: 1 if n > .5 else -1)
