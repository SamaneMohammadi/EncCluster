import torch
import collections
import flwr as fl

class Client(fl.client.NumPyClient):

	def __init__(self, cid, num_clients, model_loader, model_name, data_loader, batch_size=128, data_name='cifar10',
			input_res=32, skewness=10.0, seed=42, use_cutmix=False, device='cuda'):
		self.cid = cid
		self.use_cutmix = use_cutmix
		(self.data , self.num_classes, self.num_samples) = data_loader(id=cid, num_clients=num_clients, model_name=model_name,
			name=data_name, skewness=skewness, use_cutmix=self.use_cutmix, batch_size=batch_size, input_res=input_res, seed=seed)
		self.input_res = input_res
		self.model_name = model_name
		self.model_loader = model_loader
		self.device = device

	def set_parameters(self, parameters, config):
		if not hasattr(self, 'model'):
			self.model = self.model_loader(model=self.model_name, num_classes=self.num_classes).to(self.device)
		params_dict = zip((n for n, _ in self.model.get_params()), parameters)
		self.model.load_params(collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict}))

	def get_parameters(self, config={}):
		return [p.cpu().detach().numpy() for _,p in self.model.get_params()]

	def fit(self, parameters, config):
		self.set_parameters(parameters, config)
		self.model.compile(lr=config['lr'])
		metrics = self.model.fit(ds=self.data, epochs=config['epochs'], cutmix=self.use_cutmix, verbose=False)
		params = self.get_parameters()
		del self.model; torch.cuda.empty_cache();
		return params, self.num_samples, metrics

	def evaluate(self, parameters, config):
		raise NotImplementedError('Client-side evaluation is not implemented!')

