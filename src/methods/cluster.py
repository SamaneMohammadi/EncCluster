"""Cluster-only baseline: weight clustering, expand, plaintext weighted average
(isolates the effect of clustering compression). BN buffers averaged in plaintext."""

from clustering import weight_clustering, unflatten_state, aggregate_buffers


def aggregate(states, sizes, kappa=16, **_):
    total, expanded, keys, shapes = sum(sizes), [], None, None
    for sd in states:
        Z, P, keys, shapes = weight_clustering(sd, kappa=kappa)
        expanded.append(Z[P])
    agg = sum(v * (w / total) for v, w in zip(expanded, sizes))
    out = unflatten_state(agg, keys, shapes)
    out.update(aggregate_buffers(states, sizes))
    return out
