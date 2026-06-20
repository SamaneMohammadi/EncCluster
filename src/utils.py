import time
import logging
import itertools
import collections
import numpy as np
from sklearn.cluster import KMeans
from pyfilters import Fuse16 as Filter

" Initialize logger for logging experiment results."
def init_logger(fp='./flwr_log.log', log_lvl=logging.INFO):
	logger = logging.getLogger('flwr')
	logger.setLevel(log_lvl)
	file_log = logging.FileHandler(fp, mode='w')
	file_log.setFormatter(logging.Formatter("flwr | %(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"))
	logger.addHandler(file_log)
	return logger

"Apply weight clustering on model weights."
def weight_clustering(weights, num_clusters=64, seed=0):
	kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=1000, n_init="auto", random_state=seed)
	kmeans.fit(weights.astype('double'))
	return [kmeans.cluster_centers_.flatten(), kmeans.labels_.flatten()]

"Reshape flattened weights to original shapes."
def reshape_weights(flattened_weights, shapes):
	assert len(flattened_weights) == sum(np.prod(shape) for shape in shapes), 'Shapes do not match flattened weights!'
	reshaped_weights = []
	start = 0
	for shape in shapes:
		size = np.prod(shape) if shape else 0
		if size > 0:
			weight = np.reshape(flattened_weights[start:start+size], shape)
			reshaped_weights.append(weight)
		elif size == 0:
			reshaped_weights.append(np.array(0.0))
		start += size
	return reshaped_weights

"Reconstruct weights from cluster centers and cluster indexes."
def reconstruct_weights(cluster_centers, cluster_indexes, shapes=None):
	cluster_indexes = list(map(int, cluster_indexes))
	_weights =  [cluster_centers[idx] for idx in cluster_indexes]
	if shapes is not None:
		_weights = reshape_weights(_weights, shapes)
	return _weights

"Populate dictionary with (cluster,indexes) pairs from list of indexes."
def lst2dict(idxs_list):
	result = {}
	for i,idx in enumerate(idxs_list):
		if idx not in result:
			result[idx] = []
		result[idx].append(i)
	return result

"Populate list of indexes from dictionary with (cluster,indexes) pairs."
def dict2lst(idxs_dict):
	max_idx = max(max(i) for i in idxs_dict.values())
	arr = [None] * (max_idx + 1)
	for idx, indices in idxs_dict.items():
		for i in indices:
			arr[i] = idx
	if any(x is None for x in arr):
		raise ValueError("Some indices are missing from the input dictionary.")
	return arr

"Revert list to dictionary with (cluster,indexes) pairs."
def reverse(key_index_list):
	clusters_idxs = collections.defaultdict(list)
	for key_index in key_index_list:
		key, index = key_index.split('_')
		clusters_idxs[key].append(int(index))
	return clusters_idxs

"Create binary fuse filter from list of indexes."
def populate_filter(indexes, cid):
	# Create (cluster,indexes) pairs
	clusters_idxs = lst2dict(indexes)
	# Create entries
	v = list(itertools.chain.from_iterable([f"{k}_{i}" for i in values] for k, values in clusters_idxs.items()))
	# Create filter
	_filter = Filter(size=len(v))
	# Insert entries to filter
	s_time = time.time()
	_filter.populate(v)
	exec_time = time.time() - s_time
	# Store necessary information to reconstruct filter on server-side.
	fingerprints = _filter.fingerprints.tolist()
	comm_msg = {
		'num_elements':len(v),
		'seed': _filter.seed,
		'fingerprints': fingerprints,
	}
	transmitted_bits=64+32+16*len(fingerprints)
	return (comm_msg, cid, transmitted_bits, exec_time)

"Reconstruct indexes from binary fuse filters."
def reconstruct_indexes(filter_info, cid, num_params, num_clusters=64, true_idxs=[]):
	# Search space
	query_space = [f"{k}_{i}" for k,i in itertools.product(np.arange(num_clusters), np.arange(num_params))]
	# Re-create filter
	_filter = Filter(size=filter_info['num_elements'])
	_filter.modify(entries={'seed': filter_info['seed'], 'fingerprints': filter_info['fingerprints']})
	# Extract estimated indexes by set membership query
	s_time = time.time()
	indexes = _filter.query(query_space)
	exec_time = time.time() - s_time
	# Extract stats
	true_idx = set(list(itertools.chain.from_iterable([f"{k}_{i}" for i in values] for k, values in lst2dict(true_idxs).items())))
	fps = len([x for x in indexes if x not in true_idx])
	# Format back to list of indexes
	indexes = dict2lst(reverse(indexes))
	return (indexes, cid, fps, exec_time)

