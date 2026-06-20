import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import flwr as fl
import argparse
from utils import init_logger

parser = argparse.ArgumentParser(description="Federated Learning Parameters")
parser.add_argument("--num_rounds",				default=20,					type=int,		help="number of federated rounds")
parser.add_argument("--num_clients",			default=10,					type=int,		help="number of clients")
parser.add_argument("--num_epochs",				default=4,					type=int,		help="number of local train epochs")
parser.add_argument("--num_clusters",			default=128,				type=int,		help="number of clsuter used in weight clustering")
parser.add_argument("--init_cluster_rnd",		default=2,					type=int,		help="round to start weight clustering")
parser.add_argument("--gRAM_per_client",		default=0.066,				type=float,		help="gpu memory utilization per client instance")
parser.add_argument("--cpu_cores_per_client",	default=6,					type=int,		help="number of cpu cores allowed per clients instance")
parser.add_argument("--model",					default='resnet20',			type=str,		help="model to be used")
parser.add_argument("--dataset",				default='cifar10',			type=str,		help="dataset to be used")
parser.add_argument("--use_cutmix",				action='store_true', 						help="Use cutmix augmentation")
parser.add_argument("--skewness",				default=10000.,				type=float,		help="class distribution skewness")
parser.add_argument("--participation",			default=1.,					type=float,		help="participation percentage of clients in each round")
parser.add_argument("--batch_size",				default=128,				type=int,		help="data batch size")
parser.add_argument("--input_res",				default=32,					type=int,		help="input image resolution")
parser.add_argument("--lr",						default=1e-3,				type=float,		help="learning rate")
parser.add_argument("--timeout",				default=1000,				type=int,		help="maximum seconds of federated round until timeout")
parser.add_argument("--device",					default='cuda',				type=str,		help="device to utilize")
parser.add_argument("--log_dir",				default='../../../assets',	type=str,		help="logs store dir")
parser.add_argument("--seed",					default=42,					type=int,		help="reproducability seed")
args = parser.parse_args()

# Create log dir (if needed)
os.makedirs(f"{os.path.join(os.path.dirname(os.path.abspath(__file__)),args.log_dir)}", exist_ok=True)
logger_name = f'{os.path.dirname(os.path.abspath(__file__))}/{args.log_dir}/{os.path.basename(os.path.dirname(os.path.abspath(__file__)))}_{args.model}_{args.dataset}_{"iid" if args.skewness>10 else "niid"}_{int(100*args.participation)}_{args.num_rounds}'

def client_fn(cid):
	import os, sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
	from network import Model
	from client import Client
	from data import load_data
	return Client(int(cid), num_clients=args.num_clients, model_loader=Model, model_name=args.model,
		data_loader=load_data, device=args.device, input_res=args.input_res, batch_size=args.batch_size,
		use_cutmix=True, data_name=args.dataset, skewness=args.skewness, seed=args.seed)

def server_fn():
	from network import Model
	from server import Server
	from data import load_data
	return Server(num_rounds=args.num_rounds, num_clients=args.num_clients, participation=args.participation,
		model_loader=Model, model_name=args.model, input_res=args.input_res, data_loader=load_data, batch_size=args.batch_size,
		data_name=args.dataset, num_epochs=args.num_epochs, num_clusters=args.num_clusters, lr=args.lr, model_fp=logger_name,
		init_cluster_rnd=args.init_cluster_rnd)

def main():
	# Create server
	server = server_fn()
	# Start simulation
	history = fl.simulation.start_simulation(client_fn=client_fn, server=server, num_clients=args.num_clients,
		ray_init_args= {"ignore_reinit_error": True, "include_dashboard": False,},
		client_resources = {"num_cpus": int(args.cpu_cores_per_client), "num_gpus": float(args.gRAM_per_client),},
		config=fl.server.ServerConfig(num_rounds=args.num_rounds, round_timeout=args.timeout),)
	return history

if __name__ == "__main__":
	parsed_args = 'Parameters:\n ' + ' '.join(f'{k}={v}\n' for k, v in vars(args).items()) 
	logger = init_logger(fp=f"{logger_name}.log")
	logger.info(parsed_args)
	print(main())
