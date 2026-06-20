# EncCluster

Implementation of **EncCluster: Scalable Functional Encryption in
Federated Learning through Weight Clustering and Probabilistic Filters**.

📄 Paper: [Pervasive and Mobile Computing 108 (2025) 102021](https://www.sciencedirect.com/science/article/pii/S1574119225000100) · [arXiv:2406.09152](https://arxiv.org/abs/2406.09152)

EncCluster is a secure aggregation method for federated learning. Clients
compress their model update into a small set of weight-cluster centroids,
encrypt only those centroids with decentralized functional encryption, and encode
the cluster-to-weight mapping into a Binary Fuse filter. The server reconstructs
the mapping and aggregates the encrypted centroids directly — protecting against
inference attacks while keeping communication below FedAvg and encryption costs
low enough for edge devices.

## Methods

Select with `--method`:

- **`enccluster`** — the full method: weight clustering + Binary Fuse filter +
  DMCFE secure weighted aggregation over encrypted centroids.
- **`bfuse`** — clustering + Binary Fuse filter, plaintext aggregation.
- **`cluster`** — weight clustering only, plaintext aggregation.
- **`fedavg`** — standard FedAvg.

Models (`--model`): `resnet20`, `mlpmixer`, `convnext`.

## Setup

```bash
pip install -r requirements.txt
./setup_modules.sh        # builds the Binary Fuse filter and installs the DMCFE backend
```

## Usage

```bash
cd src
python train.py --method enccluster --model resnet20 --rounds 10
python train.py --method bfuse      --model mlpmixer --rounds 50
python train.py --method fedavg     --model convnext --rounds 50
```

CIFAR-10 downloads automatically. Functional-encryption decryption runs per
parameter, so `enccluster` suits smaller models; the baselines scale to the
larger architectures.

## Citation

```bibtex
@article{tsouvalas2025enccluster,
  title   = {EncCluster: Scalable Functional Encryption in Federated Learning
             through Weight Clustering and Probabilistic Filters},
  author  = {Tsouvalas, Vasileios and Mohammadi, Samaneh and Balador, Ali and
             Ozcelebi, Tanir and Flammini, Francesco and Meratnia, Nirvana},
  journal = {Pervasive and Mobile Computing},
  volume  = {108},
  pages   = {102021},
  year    = {2025}
}
```

## License

MIT. The vendored `mife` (PyMIFE) and `pyfilters` (Binary Fuse filter) libraries
retain their own licenses.
