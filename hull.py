import numpy as np


def in_hull_l1(p, simplex_min, simplex_max, threshold):
    K = len(simplex_min)
    for d in range(K):
        if (p[d] < (simplex_min[d] - threshold)) or (p[d] > (simplex_max[d] + threshold)):
            return False
    return True


def get_hull_l1(base, emb, threshold=0.):
    simplex_min = emb[base].min(axis=0) - threshold
    simplex_max = emb[base].max(axis=0) + threshold

    mask = (simplex_min <= emb) * (emb <= simplex_max)
    mask = np.all(mask, axis=1)
    res = np.argwhere(mask).reshape(-1)
    return res
