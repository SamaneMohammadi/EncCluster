"""EncCluster (the full method): clustering + Binary Fuse filter encoding
+ DMCFE secure weighted aggregation over encrypted centroids.

Clients encrypt only their kappa weight-centroids and encode the cluster-weight
mapping into a BF filter. The server reconstructs each mapping from the filter
and securely aggregates the encrypted centroids without decrypting any update.
BatchNorm buffers (not weights) are aggregated in plaintext."""

import numpy as np
from clustering import weight_clustering, unflatten_state, aggregate_buffers
from filters import encode_mapping, reconstruct_mapping


def aggregate(states, sizes, dmcfe=None, tag=b"round", kappa=16, bpe=16, **_):
    # cluster every client first, so the precision can be sized to the centroids
    clustered = [weight_clustering(sd, kappa=kappa) for sd in states]
    max_abs = max(float(np.max(np.abs(Z))) for Z, _, _, _ in clustered)
    dmcfe.set_precision(sum(sizes), max_abs)

    ciphertexts, filters = [], []
    for cid, (Z, P, keys, shapes) in enumerate(clustered):
        ciphertexts.append(dmcfe.encrypt_centroids(cid, Z, tag))
        filters.append(encode_mapping(P, bpe=bpe))

    d = len(clustered[0][1]); keys, shapes = clustered[0][2], clustered[0][3]
    mappings = [reconstruct_mapping(bf, d, kappa) for bf in filters]
    flat = dmcfe.secure_aggregate(ciphertexts, mappings, sizes, tag, d)
    out = unflatten_state(np.asarray(flat), keys, shapes)
    out.update(aggregate_buffers(states, sizes))
    return out
