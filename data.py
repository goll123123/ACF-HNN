#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 21:28:56 2023

@author: amax
"""

import os, inspect, random, pickle
import numpy as np, scipy.sparse as sp
from tqdm import tqdm
import pickle, torch
import torch.nn.functional as F
import math
from itertools import product
import sys
import pickle as pkl
import networkx as nx
from collections import defaultdict
sys.setrecursionlimit(99999)

def euclidean_distance(vec1, vec2):
    """
    :param vec1:
    :param vec2:
    :return:
    """
    # print('vec1', torch.tensor(vec1).size())
    # print('vec2', torch.tensor(vec2).size())
    # return F.cosine_similarity(torch.tensor(vec1).unsqueeze(0), torch.tensor(vec2).unsqueeze(0)).item()
    # return np.sqrt(np.sum(np.square(vec1 - vec2)))
    # return sum([(x - y) ** 2 for (x, y) in zip(vec1, vec2)]) ** 0.5
    return np.linalg.norm(vec1 - vec2, ord=2)

def load(dataset, run_i):
    """
    parses the dataset
    """
    # LoadData
    # if dataset == 'citeseer' or 'pumbed': 
    #     adj, features, labels, label, idx_train, idx_val, idx_test = load_data_new(
    #         dataset, run_i)
        
    # else:
    root = '/data0/wy/graph_nets-master/WebKB/'
    data = torch.load(root + dataset + "/processed/data.pt")
    data = data[0]#.to(device)
    
    edge_index = data.edge_index.t().tolist()
    neighbors = {}
    idx = [i for i in range(len(data.y))]
    for j in range(len(idx)):
        neighbors[idx[j]] = []
        for i in range(len(edge_index)):    
            if idx[j] in edge_index[i]:
                neighbors[idx[j]] += [edge_index[i][1 - edge_index[i].index(idx[j])]]
        neighbors[idx[j]] = list(set(neighbors[idx[j]]))
    
    adj = torch.zeros(len(data.y), len(data.y))#.to(device)
    for i in range(adj.size(0)):
        lines = neighbors[i]
        for j in range(len(lines)):
            adj[i][lines[j]] = 1
    
    adj = torch.mul((torch.ones(adj.size())-torch.eye(adj.size(0))),adj)
    features, labels = np.array(data.x), np.array(F.one_hot(data.y))
    label = data.y

    idx_train = torch.nonzero(data.train_mask.t()[run_i]).squeeze(1)
    idx_val = torch.nonzero(data.val_mask.t()[run_i]).squeeze(1)
    idx_test = torch.nonzero(data.test_mask.t()[run_i]).squeeze(1)
        
    train = idx_train.tolist()
    val = idx_val.tolist()
    test = idx_test.tolist()
    
    # train = torch.nonzero(data.train_mask.t()[i]).tolist()
    # val = torch.nonzero(data.val_mask.t()[i]).tolist()
    # test = torch.nonzero(data.test_mask.t()[i]).tolist()
    
    # neighbors_train = {}
    # idx = [i for i in range(len(data.y))]
    # for j in range(len(idx)):
    #     neighbors_train[idx[j]] = []
    #     for i in range(len(edge_index)):    
    #         if idx[j] in edge_index[i] and idx[j] in train and edge_index[i][1 - edge_index[i].index(idx[j])] in train:
    #             neighbors_train[idx[j]] += [edge_index[i][1 - edge_index[i].index(idx[j])]]
    #     neighbors_train[idx[j]] = list(set(neighbors_train[idx[j]]))
    
    # neighbors_label = {}
    # for j in range(len(idx)):
    #     neighbors_label[idx[j]] = []
    #     for k in range(len(train)):
    #         if data.y[train[k]] == idx[j]:
    #             neighbors_label[idx[j]].append(train[k])
    
    neighbors_train = {}
    idx = [i for i in range(labels.shape[1])]
    for j in range(len(idx)):
        neighbors_train[idx[j]] = []
    for j in range(len(train)):
        neighbors_train[idx[label[train[j]]]].append(train[j])
        
    ancor = np.array([np.mean(features[neighbors_train[j]],0) for j in range(len(neighbors_train))])
    one = np.ones_like(ancor[0])
    for j in range(len(idx)):
        if len(neighbors_train[j]) == 0:
            ancor[j] = one
    
    ## 1.按训练集标签
    # hypergraph = {} 
    # idx = [j for j in range(labels.shape[1]+1)]
    # for j in range(len(idx)-1):
    #     hypergraph[idx[j]] = []
    #     for k in range(len(train)):
    #         if data.y[train[k]] == idx[j]:
    #             hypergraph[idx[j]].append(train[k])
    # hypergraph[idx[-1]] = val + test
    
    ## 2.聚类算法(可能出现空集)
    # hypergraph = {} 
    # for j in range(len(idx)):
    #     hypergraph[idx[j]] = []
    # for j in range(len(features)):
    #     # features_j = features[j].reshape(1,len(features[j])).repeat(len(idx),axis=0)
    #     plabel = np.argmax([euclidean_distance(features[j],ancor[k]) for k in range(len(ancor))])
    #     hypergraph[plabel].append(j)
    
    ## 3.聚类算法(kNN,有放回采样)
    # n = math.floor(features.shape[0]/labels.shape[1])
    # hypergraph = {} 
    # for j in range(len(idx)):
    #     plabel = np.array([euclidean_distance(features[k],ancor[j]) for k in range(len(features))]).argsort()[-n:][::-1]
    #     hypergraph[idx[j]]=list(plabel)
    
    ## 4.为同配节点构建高阶关系
    # hypergraph = {} 
    # idx = [j for j in range(labels.shape[1])]
    # for j in range(len(idx)):
    #     hypergraph[idx[j]] = []
    # for j in range(len(features)):
    #     label_j = data.y[j]
    #     if len(neighbors[j]) != 0 and len(neighbors_train[j]) != 0:
    #         ho = (torch.nonzero(data.y[neighbors_train[j]]==label_j).size(0) + 1)/len(neighbors_train[j])
    #     elif len(neighbors[j]) == 0:
    #         ho = 1
    #     else:
    #         ho = 0       
    #     if ho > 0.5:
    #         hypergraph[data.y[j].item()].append(j)
    
    # for j in range(len(hypergraph)):
    #     if len(hypergraph[j]) == 0:
    #         allnodes = torch.nonzero(data.y[train]==j)
    #         if allnodes.size(0) == 1:
    #             hypergraph[j] += allnodes.tolist()[0]
    #         else:
    #             hypergraph[j] += allnodes.squeeze().tolist()
    
    ## 5.按与类原型相似度构建高阶关系
    # N, C, d = features.shape[0], labels.shape[1], features.shape[1]
    # hypergraph = {} 
    # idx = [j for j in range(C)]
    # All_dis = []
    # for j in range(N):
    #     # features[j].repeat(C).reshape(C,d)
    #     sim = [euclidean_distance(ancor[k], features[j]) for k in range(C)]
    #     All_dis.append(sim)
    # All_dis = np.array(All_dis)
    
    # # dis = np.mean(All_dis)
    # dis = 2*np.min(All_dis)
    
    # one = np.ones_like(All_dis)
    # zero = np.zeros_like(All_dis)
    # All_dis = np.where(All_dis > dis, zero, one)
    
    # for j in range(C):
    #     hypergraph[idx[j]] = []
    #     hypergraph[idx[j]] += np.nonzero(All_dis.transpose()[j])[0].tolist()
    
    # for j in range(C):
    #     if len(hypergraph[idx[j]]) == 0:
    #         hypergraph[idx[j]] = [0]
    
    ## 6.
    hypergraph = {} 
    
    dataset = {'hypergraph': hypergraph, 'features': features, 'labels': labels, 'n': features.shape[0],
               'neighbors_train': neighbors_train, 'N': features.shape[0], 'C': labels.shape[1],
               'd': features.shape[1], 'label': label}    

    return dataset, idx_train, idx_val, idx_test

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data_new(dataset_str, split):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    
    root = '/data0/wy/GloGNN-master/'
    # root = '/home/amax/wy/datasets/heterophily_node_classification/'
    
    # print('dataset_str', dataset_str)
    # print('split', split)
    if dataset_str in ['citeseer', 'cora', 'pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(root + "small-scale/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            root + "small-scale/data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        splits_file_path = root + 'small-scale/splits/' + dataset_str + \
            '_split_0.6_0.2_' + str(split) + '.npz'
            
        # split_new
        # splits_file_path = '/data0/wy/graph_nets-master/WebKB_SSNC/' + dataset_str + '/raw/' + \
        #     dataset_str + '_split_0.6_0.2_' + str(split) + '.npz'

        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        idx_train = list(np.where(train_mask == 1)[0])
        idx_val = list(np.where(val_mask == 1)[0])
        idx_test = list(np.where(test_mask == 1)[0])

        no_label_nodes = []
        if dataset_str == 'citeseer':  # citeseer has some data with no label
            for i in range(len(labels)):
                if sum(labels[i]) < 1:
                    labels[i][0] = 1
                    no_label_nodes.append(i)

            for n in no_label_nodes:  # remove unlabel nodes from train/val/test
                if n in idx_train:
                    idx_train.remove(n)
                if n in idx_val:
                    idx_val.remove(n)
                if n in idx_test:
                    idx_test.remove(n)

    elif dataset_str in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        graph_adjacency_list_file_path = os.path.join(
            root + 'small-scale/new_data', dataset_str, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join(root + 'small-scale/new_data', dataset_str,
                                                                f'out1_node_feature_label.txt')
        # graph_adjacency_list_file_path = os.path.join(
        #     '/data0/wy/graph_nets-master/WebKB_SSNC/', dataset_str, '/raw/out1_graph_edges.txt')
        # graph_node_features_and_labels_file_path = os.path.join('/data0/wy/graph_nets-master/WebKB_SSNC', dataset_str,
        #                                                         f'/raw/out1_node_feature_label.txt')
        graph_dict = defaultdict(list)
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                graph_dict[int(line[0])].append(int(line[1]))
                graph_dict[int(line[1])].append(int(line[0]))

        # print(sorted(graph_dict))
        graph_dict_ordered = defaultdict(list)
        for key in sorted(graph_dict):
            graph_dict_ordered[key] = graph_dict[key]
            graph_dict_ordered[key].sort()

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict_ordered))
        # adj = sp.csr_matrix(adj)

        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_str == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(
                        line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        features_list = []
        for key in sorted(graph_node_features_dict):
            features_list.append(graph_node_features_dict[key])
        features = np.vstack(features_list)
        features = sp.csr_matrix(features)

        labels_list = []
        for key in sorted(graph_labels_dict):
            labels_list.append(graph_labels_dict[key])

        label_classes = max(labels_list) + 1
        labels = np.eye(label_classes)[labels_list]

        splits_file_path = root + 'small-scale/split_new/' + dataset_str + \
            '_split_0.6_0.2_' + str(split) + '.npz'
        # split_new, splits
            
        # splits_file_path = '/data0/wy/graph_nets-master/WebKB_SSNC/' + dataset_str + '/raw' + \
        #     dataset_str + '_split_0.6_0.2_' + str(split) + '.npz'

        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        idx_train = np.where(train_mask == 1)[0]
        idx_val = np.where(val_mask == 1)[0]
        idx_test = np.where(test_mask == 1)[0]

    # adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    # features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels))[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    label = labels
    features, labels = np.array(features), np.array(F.one_hot(labels))
    
    # print('adj', adj.shape)
    # print('features', features.shape)
    # print('labels', labels.shape)
    # print('idx_train', idx_train.shape)
    # print('idx_val', idx_val.shape)
    # print('idx_test', idx_test.shape)
    return adj, features, labels, label, idx_train, idx_val, idx_test
