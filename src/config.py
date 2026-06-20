"""EncCluster settings. Defaults follow the paper; the demo uses a small model."""

# Federated
NUM_CLIENTS = 5
NUM_ROUNDS = 100
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.01
BATCH_SIZE = 64
NUM_CLASSES = 10

# Weight clustering
KAPPA = 16                 # centroids per client (paper sweeps {16..256})

# Binary Fuse filter
BPE = 8                    # bits-per-entry; 8 -> ~2^-8 FP/query, 16 -> ~2^-16

# DMCFE
FE_PRECISION = 5           # fixed-point digits for float->int
DECRYPT_BOUND = (-10_000_000, 10_000_000)

SEED = 42
