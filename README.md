# EncCluster

Implementation of **EncCluster: Scalable Functional Encryption in Federated
Learning through Weight Clustering and Probabilistic Filters**.

📄 Paper: [Pervasive and Mobile Computing 108 (2025) 102021](https://www.sciencedirect.com/science/article/pii/S1574119225000100) · [arXiv:2406.09152](https://arxiv.org/abs/2406.09152)

EncCluster offers robust privacy protection against inference attacks while
requiring minimal communication and computation overhead for clients
participating in FL. The framework is built on three building blocks: (i) model
compression via weight clustering, (ii) decentralized FE, allowing cryptographic
encryption without a fully trusted third party, and (iii) encoding via a
probabilistic data structure, termed Binary Fuse (BF) filters, to enhance privacy
without introducing excessive computational burdens. Weight clustering is applied
locally on clients' models, and the resulting set of cluster centroids is
encrypted via FE. The cluster-weight mapping, which signifies associations
between positions in the weight matrix and respective centroids, is then injected
into BF filters through computationally efficient hashing operations. To fuse all
model updates, the server reconstructs this mapping via a membership query in the
BF filters and performs a secure aggregation without decrypting the clients'
model updates. In doing so, EncCluster restricts the computationally "heavy"
encryption operations to a small set of centroid values, while their mapping to
model weights is encoded through cost-effective hashing operations, striking a
balance between preserving privacy and meeting practical computational and
communication demands in FL.

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
  title={EncCluster: Scalable functional encryption in federated learning through weight clustering and probabilistic filters},
  author={Tsouvalas, Vasileios and Mohammadi, Samaneh and Balador, Ali and Ozcelebi, Tanir and Flammini, Francesco and Meratnia, Nirvana},
  journal={Pervasive and Mobile Computing},
  volume={108},
  pages={102021},
  year={2025},
  publisher={Elsevier}
}

```

## License

MIT. The vendored `mife` (PyMIFE) and `pyfilters` (Binary Fuse filter) libraries
retain their own licenses.
