"""DMCFE weighted secure aggregation over encrypted centroids (Eq. secure_agg),
built on mife's multi-client inner-product scheme (eprint 2017/989).

Each client encrypts its kappa centroids. For parameter i the server gathers the
clients' encrypted centroids Zhat_n[P_n[i]] and decrypts the weighted sum
  sum_n |D_n| * Z_n[P_n[i]],
then divides by |D| -> weighted FedAvg, computed entirely on ciphertext.

mife recovers the result by a bounded discrete-log search, so the aggregated
integer must stay within DECRYPT_BOUND. We therefore set the fixed-point
precision from the total sample count, so |D| * Z * 10^prec always fits."""

import math
import numpy as np
from mife.multiclient.ddh import FeDDHMultiClient

import config

_MAX_ABS_CENTROID = 8.0   # safety margin on |centroid| for sizing the precision


class DMCFE:
    def __init__(self, num_clients, total_samples=None, bound=config.DECRYPT_BOUND):
        self.n = num_clients
        self.bound = bound
        self.key = FeDDHMultiClient.generate(num_clients, 1)   # Setup; m=1 scalar slot
        self.set_precision(total_samples if total_samples else num_clients)

    def set_precision(self, total_samples, max_abs_centroid=_MAX_ABS_CENTROID):
        """Choose prec so total_samples * max|Z| * 10^prec stays within the bound."""
        head_room = self.bound[1] / (total_samples * max(max_abs_centroid, 1e-6))
        self.prec = max(0, min(config.FE_PRECISION, int(math.log10(max(head_room, 1)))))

    def encrypt_centroids(self, client_id, centroids, tag):
        ek = self.key.get_enc_key(client_id)
        s = 10 ** self.prec
        return [FeDDHMultiClient.encrypt([int(round(float(z) * s))], tag, ek) for z in centroids]

    def secure_aggregate(self, ciphertexts, mappings, num_examples, tag, d):
        pk = self.key.get_public_key()
        sk = FeDDHMultiClient.keygen([[int(num_examples[i])] for i in range(self.n)], self.key)
        total, s = sum(num_examples), 10 ** self.prec
        mappings = [np.asarray(P, dtype=int) for P in mappings]
        out, cache = np.empty(d), {}
        for i in range(d):
            tup = tuple(int(P[i]) for P in mappings)
            if tup not in cache:
                cols = [ciphertexts[n][tup[n]] for n in range(self.n)]
                cache[tup] = (FeDDHMultiClient.decrypt(cols, tag, pk, sk, self.bound) / s) / total
            out[i] = cache[tup]
        return out
