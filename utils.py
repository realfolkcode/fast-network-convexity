from hull import get_hull_l1
import random
import torch
import numpy as np
import dgl


def read_hulls(filename):
    bases = []
    hulls = []
    with open(filename, 'r') as f:
        n = int(f.readline())
        for _ in range(n):
            bases.append(list(map(int, f.readline().split())))
        for _ in range(n):
            hulls.append(list(map(int, f.readline().split())))
    for i in range(len(bases)):
        bases[i].sort()
        hulls[i].sort()
    return bases, hulls


def comparison_score(s1, s2):
    return len(set(s1).intersection(set(s2))) / len(set(s1).union(set(s2)))


def test_comparison(emb, bases, hulls, threshold=0., log=True):
    mean_score = 0
    for i in range(len(bases)):
        hull_space = get_hull_l1(bases[i], emb, threshold)
        score = comparison_score(hull_space, hulls[i])
        mean_score += score
        if log:
            print(f'{i}: {score}')
    mean_score /= len(bases)
    return mean_score


def projection_score(h_graph, h_space):
    # h_graph and h_space need to be sorted!

    h_graph.sort()
    h_space.sort()
    s = len(h_space)
    err = 0
    s_itr = 0
    for g_itr in range(len(h_graph)):
        while s_itr < len(h_space) and h_space[s_itr] < h_graph[g_itr]:
            err += 1
            s_itr += 1
        s_itr += 1

    return (s - err) / s


def test_projection(emb, bases, hulls, threshold=0., log=True):
    mean_score = 0
    for i in range(len(bases)):
        hull_space = get_hull_l1(bases[i], emb, threshold)
        score = projection_score(hulls[i], hull_space)
        if log:
            print(f'{i}: {score}')
        mean_score += score
    mean_score /= len(bases)
    return mean_score


def set_random_seeds(seed_value=0, device='cpu'):
    '''source https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/5'''
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    dgl.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False