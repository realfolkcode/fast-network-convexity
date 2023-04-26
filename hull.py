import numpy as np


def in_hull_l1(p, simplex_min, simplex_max, threshold):
    K = len(simplex_min)
    for d in range(K):
        if (p[d] < (simplex_min[d] - threshold)) or (p[d] > (simplex_max[d] + threshold)):
            return False
    return True


def get_hull_l1(base, emb, threshold=0.):
    K = emb.shape[1]
    simplex_min = np.ones(K) * emb[base[0]]
    simplex_max = np.ones(K) * emb[base[0]]
    for d in range(K):
        for i in range(len(base)):
            if emb[base[i], d] < simplex_min[d]:
                simplex_min[d] = emb[base[i], d]
            if emb[base[i], d] > simplex_max[d]:
                simplex_max[d] = emb[base[i], d]

    res = []
    for i in range(emb.shape[0]):
        if in_hull_l1(emb[i], simplex_min, simplex_max, threshold):
            res.append(i)
    return res