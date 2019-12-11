# coding=utf-8
"""
@Description:
@Author: Lingyong Yan
@Date: 2019-07-25 02:51:02
@LastEditTime: 2019-08-15 01:13:47
@LastEditors: Lingyong Yan
@Python release: 3.7
@Notes:
"""
import torch


def sparse_softmax(sparse_mat, steps):
    """
    @description: return softmax of a sparse matrix
    @param {type}
    @return:
    """
    values = sparse_mat._values()
    assert sparse_mat.size(0) == steps.size(0)
    start = 0
    for step in steps:
        end = start + step
        vals = torch.nn.functional.softmax(values[start:end], dim=-1)
        values[start:end] = vals
        start = end
    return sparse_mat


def sparse_softmax_eff(sparse_mat):
    """
    @description: return softmax of a sparse matrix
                  in efficient way.
    @param {type}
    @return:
    """
    sparse_sum = torch.sparse.sum(sparse_mat, dim=-1)
    dense_sum = 1. / (sparse_sum.to_dense() + 1e-32)
    dense_sum = dense_sum.unsqueeze(-1).repeat(1, sparse_mat.size(1))
    sparse_sum = dense_sum.sparse_mask(sparse_mat)
    sparse_prob = sparse_mat * sparse_sum
    return sparse_prob


def sparse_cosine_sim(a, b, dim=-1, eps=1e-8):
    """calculate the cosine similarity and avoid the zero-division
    """
    a_norm = a / (a.norm(dim=dim)[:, None]).clamp(min=eps)
    b_norm = b / (b.norm(dim=dim)[:, None]).clamp(min=eps)
    if len(a.shape) <= 2:
        sim = torch.mm(a_norm, b_norm.transpose(1, 0))
    else:
        sim = torch.einsum('ijk, lmk->iljm', (a_norm, b_norm))
    return sim
