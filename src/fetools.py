"""Functional-encryption tools for EncCluster's secure aggregation.

Wraps the DMCFE multi-client inner-product scheme (mife FeDDHMultiClient) to
securely compute a weighted average of clients' (reconstructed) model weights.
Floats are encoded to integers with a fixed-point scale before encryption.

The weighted FedAvg is carried in the functional key (v_n = |D_n|): for a weight
position, the server decrypts  sum_n |D_n| * w_{n}  and divides by |D|. Because
the scheme recovers the result by a bounded discrete-log search, the precision is
adapted per position to keep the aggregate within BOUND, and identical
cross-client value tuples are decrypted once and cached (positions that map to
the same centroids across all clients share a result).
"""

from mife.multiclient.ddh import FeDDHMultiClient
from mife.data.fastecdsa_wrapper import WrapCurve
from fastecdsa.curve import P192 as Curve

BOUND = (-10_000_000, 10_000_000)
_PREC = 5                       # fixed-point digits when values are small
MK, MKP, CID_KEYS = None, None, {}


def keygen(num_clients):
    """[Trusted setup] Generate master key, public key, and per-client enc keys."""
    global MK, MKP, CID_KEYS
    if MK is None:
        MK = FeDDHMultiClient.generate(num_clients, 1, WrapCurve(Curve))
        MKP = MK.get_public_key()
        CID_KEYS = {cid: MK.get_enc_key(cid) for cid in range(num_clients)}


def func_key(num_clients, num_examples):
    """Functional key for weighted aggregation: v_n = |D_n| (0 if absent)."""
    v = [[num_examples[i]] if i in num_examples else [0] for i in range(num_clients)]
    return FeDDHMultiClient.keygen(v, MK)


def _prec_for(values):
    """Lower the precision when values are large so the weighted sum fits BOUND."""
    return 1 if max(abs(v) for v in values) > 10 else _PREC


def fe_aggregation(weights, num_examples, num_clients, tag=b"enccluster", bound=BOUND):
    """Securely aggregate clients' weight vectors into a weighted average.

    weights      -- list of n equal-length lists; weights[n][i] is client n's value
                    at position i (the centroid its weight i maps to).
    num_examples -- dict {cid: |D_n|}.
    Returns the length-d weighted-average vector.
    """
    keygen(num_clients)
    fkey = func_key(num_clients, num_examples)
    total = sum(num_examples.values())

    out, cache = [None] * len(weights[0]), {}
    for i, column in enumerate(zip(*weights)):          # column = each client's value at i
        key = tuple(column)
        if key not in cache:
            prec = _prec_for(column); scale = 10 ** prec
            cts = [FeDDHMultiClient.encrypt([int(round(x * scale))], tag, CID_KEYS[n])
                   for n, x in enumerate(column)]
            agg = FeDDHMultiClient.decrypt(cts, tag, MKP, fkey, bound)
            cache[key] = (agg / scale) / total
        out[i] = cache[key]
    return out
