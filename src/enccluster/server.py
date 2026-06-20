"""EncCluster server-side secure aggregation (Algorithm 2).

For the first `init_cluster_rnd` rounds the server aggregates full weights
(plain FedAvg). From then on it runs the full EncCluster path:

  1. each client sends cluster centroids + the cluster-weight mapping,
  2. the mapping is encoded into a Binary Fuse filter and reconstructed on the
     server (populate_filter / reconstruct_indexes) -- the privacy-enhancing
     probabilistic encoding,
  3. each client's weights are reconstructed from centroids + reconstructed
     mapping, and
  4. the weights are securely aggregated under decentralized functional
     encryption (fe_aggregation), giving the weighted FedAvg without the server
     ever decrypting an individual client's update.
"""

import logging
import functools
import collections

import torch
import numpy as np
import flwr as fl
from flwr.common.logger import log

from utils import reconstruct_weights, reshape_weights, populate_filter, reconstruct_indexes
from fetools import fe_aggregation, keygen


def aggregate_full_weights(results):
    """Plain weighted FedAvg over full model weights (warm-up rounds)."""
    total = sum(n for _, n, _ in results)
    weighted = [[layer * n for layer in w] for w, n, _ in results]
    return [functools.reduce(np.add, updates) / total for updates in zip(*weighted)]


def aggregate_clustered_weights(results, shapes, num_clusters, num_clients, server_round):
    """Full EncCluster aggregation: clustering + BF filter + FE secure aggregation."""
    clusters, true_indexes, num_examples, cids = zip(
        *[(r[0][0], r[0][1], r[1], r[2]) for r in results])
    num_parameters = int(sum(np.prod(s) for s in shapes))
    clusters = dict(zip(cids, clusters))
    num_examples = dict(zip(cids, num_examples))
    true_indexes = dict(zip(cids, true_indexes))

    # (1) encode each client's mapping into a Binary Fuse filter (fingerprints)
    comm_msgs = [populate_filter(true_indexes[cid], cid) for cid in cids]
    for _, cid, num_bits, _ in comm_msgs:
        log(logging.INFO, f"[Client {cid}] communicated {num_bits} bits "
                          f"(bpp: {num_bits / (32 * num_parameters):.4f}).")

    # (2) reconstruct each client's mapping from its filter (membership queries)
    filters = [m[0] for m in comm_msgs]
    recon = [reconstruct_indexes(f, cid, num_parameters, num_clusters, true_indexes[cid])
             for f, cid in zip(filters, cids)]
    est_indexes = {cid: idx for idx, cid, *_ in recon}
    total_fps = sum(fp for *_, fp, _ in recon)
    log(logging.INFO, f"[Round {server_round}] false positives: {total_fps} / "
                      f"{len(cids) * num_parameters} "
                      f"({100. * total_fps / (len(cids) * num_parameters):.4f}%).")

    # (3) reconstruct each client's weights from centroids + reconstructed mapping
    enc_data = [reconstruct_weights(clusters[cid], est_indexes[cid]) for cid in cids]

    # (4) functional-encryption secure weighted aggregation
    keygen(num_clients)
    weights_prime = fe_aggregation(enc_data, num_examples, num_clients)
    return reshape_weights(weights_prime, shapes=shapes)


class _FedAvg(fl.server.strategy.FedAvg):

    def __init__(self, shapes, init_cluster_rnd, num_clusters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shapes = shapes
        self.init_cluster_rnd = init_cluster_rnd
        self.num_clusters = num_clusters
        self.num_clients = kwargs["min_available_clients"]

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None or server_round <= 0:
            return None
        ndarrays = fl.common.parameters_to_ndarrays(parameters)
        loss, metrics = self.evaluate_fn(server_round, ndarrays, {})
        log(logging.INFO, f'[Round {server_round}] loss {loss:.3f} '
                          f'accuracy {100. * metrics["accuracy"]:.2f}')
        return loss, metrics

    def configure_fit(self, server_round, parameters, client_manager):
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        fit_ins = fl.common.FitIns(parameters, config)
        return [(c, fit_ins) for c in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results or (failures and not self.accept_failures):
            return None, {}
        weights = [(fl.common.parameters_to_ndarrays(r.parameters), r.num_examples, r.metrics["cid"])
                   for _, r in results]
        if server_round < self.init_cluster_rnd:
            aggregated = aggregate_full_weights(weights)
        else:
            aggregated = aggregate_clustered_weights(
                weights, self.shapes, self.num_clusters, self.num_clients, server_round)
        aggregated = fl.common.ndarrays_to_parameters(aggregated)
        metrics = {}
        if self.fit_metrics_aggregation_fn:
            metrics = self.fit_metrics_aggregation_fn(
                [(r.num_examples, r.metrics) for _, r in results])
        return aggregated, metrics


class Server(fl.server.Server):

    def __init__(self, model_loader, model_name, data_loader, num_rounds, num_clients=10,
                 participation=1.0, batch_size=128, data_name="cifar10", num_epochs=1, lr=1e-3,
                 num_clusters=64, input_res=32, init_cluster_rnd=2, model_fp=None, device="cuda"):
        self.num_rounds = num_rounds
        self.init_cluster_rnd = init_cluster_rnd
        self.model_fp = model_fp
        (self.ds_test, self.num_classes, self.num_samples) = data_loader(
            name=data_name, num_clients=num_clients, model_name=model_name,
            batch_size=batch_size, input_res=input_res, return_eval_ds=True)
        self.input_res = input_res
        self.model_name = model_name
        self.model_loader = model_loader
        self.clients_config = {"epochs": num_epochs, "lr": lr}
        self.num_clients = num_clients
        self.participation = participation
        self.device = device
        self.max_workers = None
        self.num_clusters = num_clusters
        self.set_strategy(self)
        self._client_manager = fl.server.client_manager.SimpleClientManager()

    def set_strategy(self, *_):
        self.strategy = _FedAvg(
            shapes=self.get_shapes(), init_cluster_rnd=self.init_cluster_rnd,
            num_clusters=self.num_clusters, min_available_clients=self.num_clients,
            fraction_fit=self.participation, min_fit_clients=int(self.participation * self.num_clients),
            fraction_evaluate=0.0, min_evaluate_clients=0, evaluate_fn=self.get_evaluation_fn(),
            on_fit_config_fn=self.get_client_config_fn(), initial_parameters=self.get_initial_parameters())

    def get_shapes(self):
        return [p.detach().numpy().shape for _, p in
                self.model_loader(model=self.model_name, num_classes=self.num_classes).get_params()]

    def set_parameters(self, parameters, config):
        if not hasattr(self, "model"):
            self.model = self.model_loader(model=self.model_name, num_classes=self.num_classes).to(self.device)
        params = zip((n for n, _ in self.model.get_params()), parameters)
        self.model.load_params(collections.OrderedDict({k: torch.tensor(v) for k, v in params}))

    def get_initial_parameters(self, *_):
        weights = [p.cpu().detach().numpy() for _, p in
                   self.model_loader(model=self.model_name, num_classes=self.num_classes).get_params()]
        return fl.common.ndarrays_to_parameters(weights)

    def get_evaluation_fn(self):
        def evaluation_fn(rnd, parameters, config):
            self.set_parameters(parameters, config)
            self.model.compile(lr=1e-3)
            loss, acc = self.model.evaluate(ds=self.ds_test, verbose=False)
            if self.model_fp:
                self.model.save_model(self.model_fp)
            del self.model; torch.cuda.empty_cache()
            return loss, {"accuracy": acc}
        return evaluation_fn

    def get_client_config_fn(self):
        def on_fit_config(rnd):
            self.clients_config["round"] = rnd
            self.clients_config["cluster_rnd"] = rnd >= self.init_cluster_rnd
            self.clients_config["num_clusters"] = self.num_clusters
            return self.clients_config
        return on_fit_config
