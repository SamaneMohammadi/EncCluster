"""BFuse baseline: clustering + Binary Fuse filter (mapping goes through encode +
reconstruct), plaintext aggregation. BN buffers averaged in plaintext."""

from clustering import weight_clustering, unflatten_state, aggregate_buffers
from filters import encode_mapping, reconstruct_mapping


def aggregate(states, sizes, kappa=16, bpe=16, **_):
    total, expanded, keys, shapes = sum(sizes), [], None, None
    for sd in states:
        Z, P, keys, shapes = weight_clustering(sd, kappa=kappa)
        P_hat = reconstruct_mapping(encode_mapping(P, bpe=bpe), len(P), kappa)
        expanded.append(Z[P_hat])
    agg = sum(v * (w / total) for v, w in zip(expanded, sizes))
    out = unflatten_state(agg, keys, shapes)
    out.update(aggregate_buffers(states, sizes))
    return out
