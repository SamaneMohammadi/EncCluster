import logging
import flwr as fl
import torch
import collections
import functools
import itertools
import numpy as np
from flwr.common.logger import log
from utils import reconstruct_weights, populate_filter, reconstruct_indexes
import concurrent.futures

def aggregate_full_weights(results):
	num_examples_total = sum([num_examples for _, num_examples,_ in results])
	weighted_weights = [[layer * num_examples for layer in weights] for weights, num_examples,_ in results]
	weights_prime = [functools.reduce(np.add, layer_updates) / num_examples_total for layer_updates in zip(*weighted_weights)]
	return weights_prime

def aggregate_clustered_weights(results, shapes, num_clusters, server_round):

	#########################################################################################
	# NOTE: Simulate compression / decompression of indexes
	clusters, true_indexes, num_examples, cids = zip(*[(result[0][0], result[0][1], result[1], result[2]) for result in results])
	num_examples_total = sum(num_examples)
	num_parameters = sum(np.prod(s) for s in shapes)
	# Convert to dict to maintain order!
	clusters = dict(zip(cids, clusters))
	num_examples = dict(zip(cids, num_examples))
	true_indexes = dict(zip(cids, true_indexes))
	#########################################################################################

	#########################################################################################
	# Compress indexes to filter
	with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
		comm_msgs = list(executor.map(populate_filter, [true_indexes[cid] for cid in cids], cids))
	#########################################################################################

	#########################################################################################
	# Measure statistics
	for (_, cid, num_bits, exec_time) in comm_msgs:
		log(logging.INFO,f"[Client {cid}] - Size of communicated message: {num_bits} bits (bbp:{num_bits/(32*8*num_parameters):.6f}). Took {exec_time:.2f} secs.")
	#########################################################################################

	#########################################################################################
	# Re-construct indexes from filter
	filters, cids, *_ = zip(*comm_msgs)
	with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
		res = list(executor.map(reconstruct_indexes, filters, cids,
			itertools.repeat(num_parameters), itertools.repeat(num_clusters), [true_indexes[cid] for cid in cids]))
	for (_,cid, fp, exec_time) in res:
		log(logging.INFO,f"[Client {cid}] - False positives: {fp} out of {num_parameters} ({100.*(fp/num_parameters):.6f}). Took {exec_time:.2f} secs.")
	est_indexes, cids, fps, exec_time = zip(*res)
	# Create dict to maintain order!
	est_indexes = dict(zip(cids, est_indexes))
	log(logging.INFO,f"[Round {server_round}] - FP: {sum(fps)} / {len(cids)*num_parameters} (fpr: {100.*sum(fps)/(len(cids)*num_parameters):.6f}). Took {sum(exec_time)/len(exec_time):.2f} secs.")
	#########################################################################################

	#########################################################################################
	# Aggregate weights
	_results = [(reconstruct_weights(clusters[cid],est_indexes[cid], shapes=shapes),num_examples[cid]) for cid in cids]
	weighted_weights = [[layer * _num_examples for layer in _weights] for (_weights,_num_examples) in _results]
	weights_prime = [functools.reduce(np.add, layer_updates) / num_examples_total for layer_updates in zip(*weighted_weights)]
	#########################################################################################
	return weights_prime

class _FedAvg(fl.server.strategy.FedAvg):

	def __init__(self, shapes, init_cluster_rnd, num_clusters, *args, **kwargs):
		super(_FedAvg, self).__init__(*args, **kwargs)
		self.shapes = shapes
		self.init_cluster_rnd = init_cluster_rnd
		self.num_clusters = num_clusters

	def evaluate(self, server_round, parameters):
		if self.evaluate_fn is None:
			return None
		# Evaluate model
		eval_res = None
		if self.evaluate_fn is not None and server_round>0:
			parameters_ndarrays = fl.common.parameters_to_ndarrays(parameters)
			eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
			log(logging.INFO,f'[Round {server_round}] - Loss: {eval_res[0]:.3f} - Accuracy: {100. * eval_res[1]["accuracy"]:.2f}')
		# Aggregate performance across tasks:
		if eval_res is None:
			return None
		return eval_res[0], eval_res[1]

	def configure_fit(self, server_round, parameters, client_manager):
		config = {}
		if self.on_fit_config_fn is not None:
			config = self.on_fit_config_fn(server_round)
		sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
		log(logging.INFO,f'[Round {server_round}] - Start of FL round with {sample_size} clients [using lr: {config["lr"]}].')
		clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
		fit_ins = fl.common.FitIns(parameters, config)
		return [(client, fit_ins) for client in clients]

	def aggregate_fit(self, server_round, results, failures,):
		if not results:
			return None, {}
		if not self.accept_failures and failures:
			return None, {}
		# Aggregate clients masks for each task seperately
		log(logging.INFO,f'[Round {server_round}] - Server-side weight aggregation starts with {len(failures)} failures.')
		weights = [(fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics['cid']) for _, fit_res in results]
		if server_round<self.init_cluster_rnd:
			parameters_aggregated = aggregate_full_weights(weights)
		else:
			parameters_aggregated = aggregate_clustered_weights(weights, self.shapes,
				num_clusters=self.num_clusters, server_round=server_round)
		# Aggregate parameters
		parameters_aggregated = fl.common.ndarrays_to_parameters(parameters_aggregated)
		# Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = {}
		if self.fit_metrics_aggregation_fn:
			fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
			metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
		return parameters_aggregated, metrics_aggregated

class Server(fl.server.Server):

	def __init__(self, model_loader, model_name, data_loader, num_rounds, num_clients=10,
		participation=1.0, batch_size=128, data_name='cifar10', num_epochs=1, lr=1e-3,
		num_clusters=64, input_res=32, init_cluster_rnd=2, model_fp=None, device='cuda'):

		self.num_rounds = num_rounds
		self.init_cluster_rnd = init_cluster_rnd
		(self.ds_test, self.num_classes, self.num_samples) = data_loader(name=data_name,
			model_name=model_name, num_clients=num_clients, batch_size=batch_size, input_res=input_res, return_eval_ds=True)
		self.input_res = input_res
		self.model_name = model_name
		self.model_loader = model_loader
		self.num_clusters = num_clusters
		self.clients_config = {"epochs":num_epochs, "lr":lr}
		self.num_clients = num_clients
		self.participation = participation
		self.device = device
		self.max_workers = None
		self.model_fp = model_fp
		self.set_strategy(self)
		self._client_manager = fl.server.client_manager.SimpleClientManager()

	def set_max_workers(self, *args, **kwargs):
		return super(Server, self).set_max_workers(*args, **kwargs)

	def set_strategy(self, *_):
		self.strategy = _FedAvg(shapes=self.get_shapes(), init_cluster_rnd=self.init_cluster_rnd,
			num_clusters=self.num_clusters, min_available_clients=self.num_clients,
			fraction_fit=self.participation, min_fit_clients=int(self.participation*self.num_clients),
			fraction_evaluate=0.0, min_evaluate_clients=0, evaluate_fn=self.get_evaluation_fn(),
			on_fit_config_fn=self.get_client_config_fn(), initial_parameters=self.get_initial_parameters(),)

	def client_manager(self, *args, **kwargs):
		return super(Server, self).client_manager(*args, **kwargs)

	def get_shapes(self):
		return [p.detach().numpy().shape for _,p in \
			self.model_loader(model=self.model_name, num_classes=self.num_classes).get_params()]

	def set_parameters(self, parameters, config):
		if not hasattr(self, 'model'):
			self.model = self.model_loader(model=self.model_name, num_classes=self.num_classes).to(self.device)
		params_dict = zip((n for n, _ in self.model.get_params()), parameters)
		self.model.load_params(collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict}))

	def get_initial_parameters(self, *_):
		init_weights = [p.cpu().detach().numpy() \
			for _,p in self.model_loader(model=self.model_name, num_classes=self.num_classes).get_params()]
		return fl.common.ndarrays_to_parameters(init_weights)

	def get_evaluation_fn(self):
		def evaluation_fn(rnd, parameters, config):
			self.set_parameters(parameters, config)
			self.model.compile(lr=1e-3)
			metrics = self.model.evaluate(ds=self.ds_test, verbose=False)
			self.model.save_model(self.model_fp)
			del self.model; torch.cuda.empty_cache();
			return metrics[0], {"accuracy":metrics[1]}
		return evaluation_fn

	def get_client_config_fn(self):
		" Define fit config function with constant self objects."
		def get_on_fit_config_fn(rnd):
			self.clients_config["round"] = rnd
			self.clients_config["cluster_rnd"] = rnd>=self.init_cluster_rnd
			self.clients_config["num_clusters"] = self.num_clusters
			return self.clients_config
		return get_on_fit_config_fn
