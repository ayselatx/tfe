import numpy as np

def fuse_concat(Xv, Xt):
    return np.concatenate([Xv, Xt], axis=1)

def fuse_add(Xv, Xt):
    d = min(Xv.shape[1], Xt.shape[1])
    return Xv[:, :d] + Xt[:, :d]

def fuse_gated(Xv, Xt, alpha=0.5):
    return alpha * Xv + (1 - alpha) * Xt

def fuse_similarity_weighted(Xv, Xt):
    sim = (Xv * Xt).sum(axis=1, keepdims=True)
    w1 = sim / (sim + 1e-8)
    w2 = 1 - w1
    return w1 * Xv + w2 * Xt
