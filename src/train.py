"""EncCluster / FedCPF training (Algorithm 1).

Each round, clients train locally; the chosen method aggregates their updates.
  enccluster -- full method: clustering + Binary Fuse filter + DMCFE secure agg
  bfuse   -- clustering + BF filter, plaintext aggregation
  cluster -- clustering only, plaintext aggregation
  fedavg  -- plain FedAvg

FE decryption is per parameter, so --method enccluster is intended for small models;
the baselines scale to the larger architectures (same result up to quantization).

    python train.py --method enccluster --model resnet20 --rounds 10
    python train.py --method fedavg --model convnext --rounds 50
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "modules", "pyfilters"))

import argparse, copy
import numpy as np
import torch

import config
from architectures import get_model
from data import get_cifar10, make_client_loaders, test_loader
from network import local_train, accuracy
from fe import DMCFE
from methods import METHODS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="enccluster", choices=list(METHODS))
    p.add_argument("--model", default="resnet20", choices=["resnet20", "mlpmixer", "convnext"])
    p.add_argument("--clients", type=int, default=config.NUM_CLIENTS)
    p.add_argument("--rounds", type=int, default=config.NUM_ROUNDS)
    p.add_argument("--kappa", type=int, default=config.KAPPA)
    p.add_argument("--bpe", type=int, default=config.BPE)
    p.add_argument("--root", default="./data_files")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(config.SEED); np.random.seed(config.SEED)
    device = args.device

    loaders, sizes = make_client_loaders(get_cifar10(args.root, True), args.clients)
    test_ld = test_loader(args.root)
    model = get_model(args.model, config.NUM_CLASSES).to(device)
    method = METHODS[args.method]
    dmcfe = DMCFE(args.clients, total_samples=sum(sizes)) if args.method == "enccluster" else None

    print(f"EncCluster | method={args.method} model={args.model} "
          f"clients={args.clients} kappa={args.kappa} bpe={args.bpe}")

    for r in range(1, args.rounds + 1):
        states = []
        for i in range(args.clients):
            m = local_train(copy.deepcopy(model).to(device), loaders[i],
                            config.LEARNING_RATE, config.LOCAL_EPOCHS, device)
            states.append({k: v.detach().clone() for k, v in m.state_dict().items()})

        new = method.aggregate(states, sizes, dmcfe=dmcfe, tag=f"round_{r}".encode(),
                               kappa=args.kappa, bpe=args.bpe)
        model.load_state_dict({k: torch.as_tensor(v) for k, v in new.items()}, strict=False)

        if r % max(1, args.rounds // 10) == 0 or r == args.rounds:
            print(f"round {r:3d} | test acc {accuracy(model, test_ld, device):.4f}")
    print("done.")


if __name__ == "__main__":
    main()
