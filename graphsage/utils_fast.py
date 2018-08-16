import pandas as pd
import networkx as nx
import json
from networkx.readwrite import json_graph
import numpy as np
import random
import sys
import os
from numba import jitclass, uint64, float32
import scipy
from scipy.sparse import csr_matrix, load_npz, save_npz, spdiags
import copy
from collections import Counter

@jitclass([
    ('K', uint64),
    ('values', uint64[:]),
    ('q', float32[:]),
    ('J', uint64[:]),
])
class FastRandomChoiceCached(object):
    def __init__(self, values, probs):
        self.values = values
        self.K = np.uint64(probs.size)
        self.q = np.zeros(self.K, dtype=np.float32)
        self.J = np.zeros(self.K, dtype=np.uint64)
        self.prep_var_sample(probs)

    def prep_var_sample(self, probs):
        smaller, larger = [], []
        for kk, prob in enumerate(probs):
            self.q[kk] = self.K * prob
            if self.q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        while len(smaller) > 0 and len(larger) > 0:
            small, large = smaller.pop(), larger.pop()
            self.J[small] = large
            self.q[large] = self.q[large] - (1.0 - self.q[small])
            if self.q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

    def draw_one(self):
        kk = int(np.floor(np.random.rand() * len(self.J)))
        if np.random.rand() < self.q[kk]:
            return self.values[kk]
        else:
            return self.values[self.J[kk]]

    def sample(self, n, r1, r2):
        res = np.zeros(n, dtype=np.int32)
        lj = len(self.J)
        for i in range(n):
            kk = int(np.floor(r1[i] * lj))
            if r2[i] < self.q[kk]:
                res[i] = kk
            else:
                res[i] = self.J[kk]
        return res

    def draw_n(self, n):
        r1, r2 = np.random.rand(n), np.random.rand(n)
        return self.values[self.sample(n, r1, r2)]

class SparseGraph:
    def __init__(self, adj_matrix=None,
                 node_features=None,
                 id_map=None,
                 class_map=None,
                 graph_weights_scaling='log2',
                 graph_mode='directed',
                 features_mode='npy',
                 features_sep=','
                 ):

        if isinstance(adj_matrix, str):
            self.load_adj(adj_matrix, weights_scaling=graph_weights_scaling, mode=graph_mode)
        elif isinstance(adj_matrix, csr_matrix):
            self.adj_matrix = adj_matrix
        else:
            raise "adj_matrix should be path to adj_matrix string or scipy.sparse.csr_matrix"
        self.adj_matrix_normed = self.normalize(self.adj_matrix)

        self.deg = np.zeros(self.adj_matrix.shape[0])
        deg = (Counter(self.adj_matrix.nonzero()[0]))
        for i in deg:
            self.deg[i] = deg[i]
        #self.deg =

        if isinstance(node_features, pd.DataFrame):
            self.features = node_features.values
        elif isinstance(node_features, np.ndarray):
            self.features = node_features
        elif isinstance(node_features, str):
            self.load_features(node_features, mode=features_mode, sep=features_sep)
        else:
            raise "nodefeatures should be np.array or pd.DataFrame or path2features"

        assert self.adj_matrix.shape[0] == self.features.shape[
            0], "adj_matrix.shape[0] should be equal to node_features.shape[0]"

        self.cache4random()

        self.id_map = id_map
        self.class_map = class_map

    def sample_neighbours(self, node_id, n_sample):
        return self.random_cached[node_id].draw_n(n_sample)

    def load_adj(self, path2matrix, weights_scaling='log2', mode='directed'):
        """"""
        self.adj_matrix = load_npz(path2matrix)
        if mode == 'undirected':
            self.adj_matrix = (self.adj_matrix + self.adj_matrix.T)

        if weights_scaling == 'log2':
            self.adj_matrix.data = np.log2(1 + self.adj_matrix.data)

    def load_features(self, path2fts, mode='csv', sep=','):
        """грузим уже обработанные фичи"""
        if mode == 'csv':
            self.features = pd.read_csv(path2fts, sep=sep)
        elif mode == 'npy':
            self.features = np.load(path2fts)

    def get_sub_graph(self, list_of_nodes):
        mask = np.zeros(self.adj_matrix.shape[0])
        mask[list_of_nodes] = 1
        sub_graph = spdiags(mask, 0, len(mask), len(mask)) * self.adj_matrix
        sub_graph = ((spdiags(mask, 0, len(mask), len(mask)) * sub_graph.T).T).tocsr()
        return sub_graph

    # @staticmethod
    # def preprocess_edgelist(path2edjlist, sep=',', graph_weights_scaling='log2', graph_mode='directed'):
    #     X = pd.read_csv(path2edjlist, sep=sep)
    #
    #     out_vertex_col = X.columns[0]
    #     in_vertex_col = X.columns[1]
    #     weight_col = X.columns[2]
    #
    #     out_vertex = set(X[out_vertex_col])
    #     in_vertex = set(X[in_vertex_col])
    #     all_vertex = out_vertex.union(in_vertex)
    #
    #     vertex_encoding = {i: j for i, j in zip(all_vertex, range(len(all_vertex)))}
    #     id_map = {j: i for i, j in zip(all_vertex, range(len(all_vertex)))}
    #
    #     X[out_vertex_col] = X[out_vertex_col].map(vertex_encoding)
    #     X[in_vertex_col] = X[in_vertex_col].map(vertex_encoding)
    #
    #     graph = csr_matrix((X[weight_col], (X[in_vertex_col], X[out_vertex_col])),
    #                        (len(vertex_encoding), len(vertex_encoding)), dtype=np.float32)
    #
    #     if graph_mode == 'undirected':
    #         graph = graph + graph.T
    #
    #     norm_consts = graph.sum(axis=1)
    #     norm_consts = np.array(norm_consts).squeeze()
    #     norm = 1 / norm_consts
    #     norm[norm == np.inf] = 0
    #     graph = spdiags(norm, 0, len(norm_consts), len(norm_consts)) * graph
    #     return graph, id_map

    def cache4random(self):

        self.random_cached = []
        for i in range(self.adj_matrix_normed.shape[0]):
            self.random_cached.append(FastRandomChoiceCached(self.adj_matrix_normed[i].indices.astype(np.uint64),
                                                             self.adj_matrix_normed[i].data))

    @staticmethod
    def normalize(graph):
        norm_consts = graph.sum(axis=1)
        norm_consts = np.array(norm_consts).squeeze()
        norm = 1 / norm_consts
        norm[norm == np.inf] = 0
        graph = spdiags(norm, 0, len(norm_consts), len(norm_consts)) * graph
        return graph

    def get_copy(self, node_list=None):
        G_copy = copy.copy(self)
        G_copy.random_cached = []
        G_copy = copy.deepcopy(G_copy)
        if node_list is not None:
            G_copy.adj_matrix = G_copy.get_sub_graph(node_list)
        G_copy.cache4random()
        return G_copy


def load_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    G_ours = nx.to_scipy_sparse_matrix(G)

    d = {'train': [], 'val': [], 'test': []}
    broken_count = 0
    for i, node in enumerate(G.nodes()):
        if G.node[node]['val']:
            d['val'].append(i)
        elif G.node[node]['test']:
            d['test'].append(i)
        else:
            d['train'].append(i)




    conversion = lambda n: int(n)
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}

    walks = []

    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)
    class_map = {conversion(k): lab_conversion(v) for k,v in class_map.items()}

    G_sup = SparseGraph(G_ours, prefix + '-feats.npy', id_map=id_map, class_map=class_map, features_mode='npy')

    G_train = G_sup.get_sub_graph(d['train'])
    print(type(G_train))
    G_sup_train = SparseGraph(G_train, prefix + '-feats.npy', id_map=id_map, class_map=class_map, features_mode='npy')

    if load_walks:
        bs = batch_sampler(scipy.sparse.load_npz(prefix+'-walks.npz'))
        return G_sup_train, G_sup, bs
    return G_sup_train, G_sup

class batch_sampler:
    def __init__(self, pairs):
        self.shape = pairs.shape
        self.node_sampler = FastRandomChoiceCached(
            np.ravel_multi_index((pairs.row, pairs.col), dims=pairs.shape).astype(np.uint64),
            pairs.data / pairs.data.sum()
        )

    def sample_batch(self, batch_size=10):
        batch = self.node_sampler.draw_n(batch_size)
        return np.unravel_index(batch, self.shape)

if __name__ == '__main__':
    G_data = json.load(open('../example_data/ppi-G.json'))
    G = json_graph.node_link_graph(G_data)
    G_ours = nx.to_scipy_sparse_matrix(G)
    G_sup = SparseGraph(G_ours, '../example_data/ppi-feats.npy', features_mode='npy')
    a = G_sup.get_copy()
    print(a.random_cached is G_sup.random_cached)
    print(len(a.random_cached))