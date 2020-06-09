import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
import random
import data.data.io as io
import os

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data_ms_academic():
    dataset = io.load_dataset('ms_academic')
    features = dataset.attr_matrix
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))
    adj = dataset.adj_matrix
    adj = adj + sp.eye(adj.shape[0])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    labels = dataset.labels

    # split by class
    split_by_class = [[] for i in range(len(dataset.class_names))]
    for i in range(len(labels)):
        split_by_class[labels[i]].append(i)

    # training set
    num_train = 20
    idx_train = np.concatenate([np.random.choice(each_class, num_train, replace=False) for each_class in split_by_class])

    # validation set
    num_val = 500
    idx_val = np.random.choice([i for i in range(dataset.attr_matrix.shape[0]) if not i in idx_train], num_val, replace=False)

    # test set
    idx_test = [i for i in range(dataset.attr_matrix.shape[0]) if (not i in idx_train) and (not i in idx_val)]

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data(dataset_str='cora'):
    print('Loading {} dataset...'.format(dataset_str))
    if dataset_str == "ms_academic":
        return load_data_ms_academic()
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty_extended[list(set(range(min(test_idx_reorder), max(test_idx_reorder) + 1)) - set(test_idx_range)) - min(
            test_idx_range), 0] = 1
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize_features(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + sp.eye(adj.shape[0])

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_split(dataset_str='cora', train_ratio=20, repeat=6):
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
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty_extended[list(set(range(min(test_idx_reorder), max(test_idx_reorder) + 1)) - set(test_idx_range)) - min(
            test_idx_range), 0] = 1
        ty = ty_extended

    ##########################################################
    # reorder because tx and ty are not ordered w.r.t. node ID
    ##########################################################
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize_features(features)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + sp.eye(adj.shape[0])

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # if train_ratio == 0:
    #     idx_test = test_idx_range.tolist()
    #     idx_train = range(len(y))
    #     idx_val = range(len(y), len(y)+500)
    # else:
    #     train_count = train_ratio * labels.shape[0] // 100
    #     idx_train = range(train_count)
    #     idx_test = []
    #     idx_val = []
    #     for i in range(train_count, labels.shape[0]):
    #         if i % 2 == 0:
    #             idx_test.append(i)
    #         else:
    #             idx_val.append(i)
    # print('#train instances:', len(idx_train))
    # print('#val instances:', len(idx_val))
    # print('#test instances:', len(idx_test))
    #
    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])

    # load flags
    with open("data/ind.{}.{}.{}.flag".format(dataset_str, str(train_ratio),
                                              str(repeat)), 'rb') as f:
        if sys.version_info > (3, 0):
            sample_flags = pkl.load(f, encoding='latin1')
        else:
            sample_flags = pkl.load(f)

    idx_train = []
    idx_val = []
    idx_test = []

    for i in range(sample_flags.shape[0]):
        if sample_flags[i] == 1:
            idx_train.append(i)
        elif sample_flags[i] == 2:
            idx_val.append(i)
        elif sample_flags[i] == 3:
            idx_test.append(i)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adj_to_graph(adj):
    indices = adj._indices().numpy()
    values = adj._values().numpy()
    num_node = adj.shape[0]
    graph = []
    for i in range(num_node):
        graph.append([])
    for edge in indices.T:
        graph[edge[0]].append(edge[1])
    return graph

def graph_sample(idx_train, batch_size, graph, depth, features, labels):
    idx = []
    for i in range(batch_size):
        idx.append(int(random.sample(list(idx_train), 1)[0]))
    y = labels[idx]
    for i in range(len(idx)):
        for j in range(depth):
            idx[i] = random.sample(graph[idx[i]], 1)[0]
    x = features[idx]
    return x, y

def graph_sample_clean(idx_train, batch_size, graph, depth, features, labels):
    idx = []
    for i in range(batch_size):
        idx.append(random.sample(idx_train, 1)[0])
    y = labels[idx]
    for i in range(len(idx)):
        for j in range(depth):
            idx[i] = random.sample(graph[idx[i]], 1)[0]
    x = features[idx]
    y_true = labels[idx]
    return x, y, y_true

def graph_sample_label(idx, num_sample, graph, depth, labels, rate = 0.0002, thre = 0.9):
    label = labels.numpy()
    num_class = np.max(label)
    sample = np.zeros((label.shape[0], num_class + 1))
    for _ in range(num_sample):
        i = random.sample(idx, 1)[0]
        c = label[i]
        for j in range(depth):
            i = random.sample(graph[i], 1)[0]
        sample[i][c] += 1
    '''
    ana = np.concatenate((sample, label[:, np.newaxis]), axis=1)
    each_num_sample = np.sum(sample, 1)
    index_sampled = np.where(each_num_sample > 0)
    rg = [[0, 50], [50, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 1000], [1000, num_sample]]
    nb_node = [0 for i in range(len(rg))]
    right_node = [0 for i in range(len(rg))]
    wrong_node = [0 for i in range(len(rg))]
    acc_node = [0 for i in range(len(rg))]
    for index in index_sampled[0]:
        for j in range(len(rg)):
            min = rg[j][0]
            max = rg[j][1]
            if each_num_sample[index] > min and each_num_sample[index] <= max and sample[index][np.argmax(sample[index])] > each_num_sample[index] * 0.8:
                nb_node[j] += 1
                if np.argmax(sample[index]) == label[index]:
                    right_node[j] += 1
                else:
                    wrong_node[j] += 1

    for i in range(len(right_node)):
        acc_node[i] = right_node[i] / nb_node[i]
    '''

    relabel = []
    reindex = []
    for i in range(label.shape[0]):
        sample_node = sample[i]
        l = np.argmax(sample_node)
        if sum(sample_node) > num_sample * rate and sample_node[l] > sum(sample_node) * thre:
            relabel.append(l)
            reindex.append(i)


    for i in idx:
        if i in reindex:
            if relabel[reindex.index(i)] != label[i]:
                relabel[reindex.index(i)] = label[i]
        else:
            reindex.append(i)
            relabel.append(label[i])

    right_nb = 0
    for i in reindex:
        if relabel[reindex.index(i)] == label[i]:
            right_nb += 1

    print('the total number of samples is %d, the accuracy of training set is %f'%(len(relabel), right_nb/len(reindex)))

    return reindex, relabel

def graph_to_pro(graph, power=0):
    graph_pro = []
    for i in graph:
        l = []
        for j in i:
            l.append(len(graph[j])**power)
        s = sum(l)
        for j in range(len(i)):
            l[j] = l[j] / s
        graph_pro.append(l)
    return graph_pro

def MC_sample(pro_dis = []):
    rdm = random.random()
    sum = 0
    for i in range(len(pro_dis)):
        sum += pro_dis[i]
        if rdm < sum:
            return  i

def graph_sample_rount(idx_all, batch_size, graph, graph_pro, depth, features, labels):
    idx = []
    for i in range(batch_size):
        idx.append(random.sample(idx_all, 1))
    for i in range(len(idx)):
        for j in range(depth):
            a = MC_sample(graph_pro[idx[i][j]])
            idx[i].append(graph[idx[i][j]][a])
    idx = np.array(idx)
    x = features[idx[:, -1]]
    y = labels[idx[:, 0]]
    y_true = labels[idx[:, -1]]
    return x, y, y_true, idx

def graph_change(route, graph, graph_pro, decay_pro):
    depth = route.shape[1]
    leaf_n = route.shape[0]
    for i in range(leaf_n):
        r = route[i]
        for j in range(depth - 1):
            pair = [r[j], r[j+1]]
            #if pair[0] == pair[1]:
            #    continue
            nei = graph[pair[0]]
            nei_pro = graph_pro[pair[0]]
            nei_n = len(nei)
            for k in range(nei_n):
                if pair[1] == nei[k]:
                    pro = nei_pro[k]
                    increase = pro * decay_pro / (nei_n - 1)
                    for x in range(nei_n):
                        if x == k:
                            nei_pro[x] -= pro * decay_pro
                        else:
                            nei_pro[x] += increase
    pass

def graph_to_adj(graph, graph_pro, adj):
    indices = adj._indices().numpy()
    s = 0
    for i, j in list(indices.T):
        nei = graph[i]
        for k in range(len(nei)):
            if nei[k] == j:
                adj._values()[s] = graph_pro[i][k]
                s = s+1
    pass

def write_file(file = 'outfile.csv', y = -10000):
    with open(file, 'a') as f:
        f.write('%.4f,' % y)

# def reduce_noise(adj, labels, rate):
#     _adj = adj.copy()
#     bad_edges = []
#     for i in range(adj.shape[0]):
#         for j in range(i+1, adj.shape[1]):
#             if labels[i].item() != labels[j].item():
#                 bad_edges.append([i, j])
#     for i_j in np.random.choice(bad_edges, int(len(bad_edges)*rate)):
#         _adj[i_j[0]][i_j[1]] = 0
#         _adj[i_j[1]][i_j[0]] = 0
#     return _adj

def reduce_noise(adj, labels, noise_rate):
    indices = adj._indices().numpy()
    upper = indices[0,:] > indices[1,:]
    upper_indices = indices[:, upper]

    bad_num = 0
    for (i, j) in np.transpose(upper_indices):
        if labels[i].item() != labels[j].item():
            bad_num += 1

    # keep_rate: the bad edge ratio left.
    keep_rate = noise_rate / ( bad_num / upper_indices.shape[1] )
    keep_flag = []
    for (i, j) in np.transpose(upper_indices):
        if labels[i].item() == labels[j].item():
            keep_flag.append(True)
        else:
            if np.random.rand(1) < keep_rate:
                keep_flag.append(True)
            else:
                keep_flag.append(False)

    upper_ind_left = upper_indices[:, keep_flag]
    node = adj.size()[0]
    ind_lef = np.concatenate((upper_ind_left, upper_ind_left[[1,0],:],
                              np.array([np.arange(0, node), np.arange(0, node)])), axis=1)

    ind_lef = torch.LongTensor(ind_lef)
    values = torch.FloatTensor(np.ones((ind_lef.shape[1])))

    _adj = torch.sparse.FloatTensor(ind_lef, values, adj.size())

    return _adj

def get_noise_rate(adj, labels):
    indices = adj._indices().numpy()
    upper = indices[0, :] > indices[1, :]
    upper_indices = indices[:, upper]

    bad_num = 0
    for (i, j) in np.transpose(upper_indices):
        if labels[i].item() != labels[j].item():
            bad_num += 1

    return bad_num / upper_indices.shape[1]