from typing import DefaultDict
from collections import defaultdict
from torch.functional import Tensor
from torch_geometric.data import Data
from utils.utilities import fixed_unigram_candidate_sampler
import torch
import numpy as np
import torch_geometric as tg
import scipy.sparse as sp


import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, args, graphs, features, adjs,  context_pairs):
        super(MyDataset, self).__init__()
        self.args = args
        self.graphs = graphs
        self.features = [self._preprocess_features(feat) for feat in features]
        self.adjs = [self._normalize_graph_gcn(a)  for a  in adjs]
        self.time_steps = args.time_steps
        self.context_pairs = context_pairs
        self.max_positive = args.neg_sample_size
        self.train_nodes = list(self.graphs[self.time_steps-1].nodes()) # all nodes in the graph.
        self.min_t = max(self.time_steps - self.args.window - 1, 0) if args.window > 0 else 0
        self.degs = self.construct_degs()
        self.pyg_graphs = self._build_pyg_graphs()
        self.__createitems__()

    def _normalize_graph_gcn(self, adj):
        """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
        rowsum = np.array(adj_.sum(1), dtype=np.float32)
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return adj_normalized

    def _preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        features = np.array(features.todense())
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def construct_degs(self):
        """ Compute node degrees in each graph snapshot."""
        # different from the original implementation
        # degree is counted using multi graph
        degs = []
        for i in range(self.min_t, self.time_steps):
            G = self.graphs[i]
            deg = []
            for nodeid in G.nodes():
                deg.append(G.degree(nodeid))
            degs.append(deg)
        return degs

    def _build_pyg_graphs(self):
        pyg_graphs = []
        for feat, adj in zip(self.features, self.adjs):
            x = torch.Tensor(feat)
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
            pyg_graphs.append(data)
        return pyg_graphs

    def __len__(self):
        return len(self.train_nodes)

    def __getitem__(self, index):
        node = self.train_nodes[index]
        return self.data_items[node]
    
    def __createitems__(self):
        self.data_items = {}
        for node in list(self.graphs[self.time_steps-1].nodes()):
            feed_dict = {}
            node_1_all_time = []
            node_2_all_time = []
            for t in range(self.min_t, self.time_steps):
                node_1 = []
                node_2 = []
                if len(self.context_pairs[t][node]) > self.max_positive:
                    node_1.extend([node]* self.max_positive)
                    node_2.extend(np.random.choice(self.context_pairs[t][node], self.max_positive, replace=False))
                else:
                    node_1.extend([node]* len(self.context_pairs[t][node]))
                    node_2.extend(self.context_pairs[t][node])
                assert len(node_1) == len(node_2)
                node_1_all_time.append(node_1)
                node_2_all_time.append(node_2)

            node_1_list = [torch.LongTensor(node) for node in node_1_all_time]
            node_2_list = [torch.LongTensor(node) for node in node_2_all_time]
            node_2_negative = []
            for t in range(len(node_2_list)):
                degree = self.degs[t]
                node_positive = node_2_list[t][:, None]
                node_negative = fixed_unigram_candidate_sampler(true_clasees=node_positive,
                                                                num_true=1,
                                                                num_sampled=self.args.neg_sample_size,
                                                                unique=False,
                                                                distortion=0.75,
                                                                unigrams=degree)
                node_2_negative.append(node_negative)
            node_2_neg_list = [torch.LongTensor(node) for node in node_2_negative]
            feed_dict['node_1']=node_1_list
            feed_dict['node_2']=node_2_list
            feed_dict['node_2_neg']=node_2_neg_list
            feed_dict["graphs"] = self.pyg_graphs
        
            self.data_items[node] = feed_dict

    @staticmethod
    def collate_fn(samples):
        batch_dict = {}
        for key in ["node_1", "node_2", "node_2_neg"]:
            data_list = []
            for sample in samples:
                data_list.append(sample[key])
            concate = []
            for t in range(len(data_list[0])):
                concate.append(torch.cat([data[t] for data in data_list]))
            batch_dict[key] = concate
        batch_dict["graphs"] = samples[0]["graphs"]
        return batch_dict


    
