import time
import argparse
import numpy as np
import torch
import scipy.sparse as sp
import warnings

def clust_rank(mat, use_ann_above_samples, initial_rank=None):
    s = mat.shape[0]

    if initial_rank is not None:
        orig_dist = torch.empty((1, 1), device=mat.device)
    elif s <= use_ann_above_samples:
        # Cosine distance
        norm_mat = torch.nn.functional.normalize(mat, p=2, dim=1)
        orig_dist = 1 - torch.mm(norm_mat, norm_mat.T)
        diag_indices = torch.arange(s, device=mat.device)
        orig_dist[diag_indices, diag_indices] = float('inf')
        initial_rank = torch.argmin(orig_dist, dim=1)

    # The Clustering Equation
    indices = torch.arange(s, device=mat.device)
    values = torch.ones(s, dtype=torch.float32, device=mat.device)
    A = torch.sparse_coo_tensor(
        torch.stack([indices, initial_rank]),
        values,
        size=(s, s)
    )
    A = A + torch.sparse_coo_tensor(
        torch.stack([indices, indices]),
        values,
        size=(s, s)
    )
    A = torch.sparse.mm(A, A.t())
    A = A.to_dense()
    A[indices, indices] = 0
    return A, orig_dist

def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        mask = (orig_dist * a) > min_sim
        a = torch.where(mask, torch.zeros_like(a), a)
    # Convert the dense matrix to a SciPy sparse matrix for connected components calculation
    a_scipy = sp.csr_matrix(a.cpu().numpy())
    num_clust, labels = sp.csgraph.connected_components(a_scipy, directed=True, connection='weak')
    labels = torch.tensor(labels, device=a.device)
    return labels, num_clust

def FINCH(data, initial_rank=None, use_ann_above_samples=70000):
    """ Simplified FINCH clustering algorithm for the first partition.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param use_ann_above_samples: Above this data size (number of samples) approximate nearest neighbors will be used to speed up neighbor
        discovery. For large scale data where exact distances are not feasible to compute, set this. [default = 70000]
    :return:
            c: Nx1 array. Cluster label for the first partition.
            num_clust: Number of clusters.
    """
    # Ensure data is a PyTorch tensor
    adj, orig_dist = clust_rank(data, use_ann_above_samples, initial_rank)
    group, num_clust = get_clust(adj, orig_dist)

    return group, num_clust