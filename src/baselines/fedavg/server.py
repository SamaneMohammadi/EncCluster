import torch
import collections
import flwr as fl

class Server(fl.server.Server):

	def __init__(self, model_loader, model_name, data_loader, num_rounds, num_clients=10,
		participation=1.0, batch_size=128, data_name='cifar10', num_epochs=1, lr=1e-3,
		input_res=32, model_fp=None, device='cuda'):

		self.num_rounds = num_rounds
		(self.ds_test, self.num_classes, self.num_samples) = data_loader(name=data_name,  num_clients=num_clients,
			model_name=model_name, batch_size=batch_size, input_res=input_res, return_eval_ds=True)
		self.input_res = input_res
		self.model_name = model_name
		self.model_loader = model_loader
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
		self.strategy = fl.server.strategy.FedAvg(
			min_available_clients=self.num_clients, fraction_fit=self.participation,
			min_fit_clients=int(self.participation*self.num_clients), fraction_evaluate=0.0,
			min_evaluate_clients=0, evaluate_fn=self.get_evaluation_fn(),
			on_fit_config_fn=self.get_client_config_fn(), initial_parameters=self.get_initial_parameters(),)

	def client_manager(self, *args, **kwargs):
		return super(Server, self).client_manager(*args, **kwargs)

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
			return self.clients_config
		return get_on_fit_config_fn
