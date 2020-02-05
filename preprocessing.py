
import os
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import scanpy as sc

flags = tf.flags
FLAGS = flags.FLAGS

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def make_log(log_str, fp):
    fp.write(log_str)
    fp.write('\n')
    print(log_str)

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def construct_feed_dict(adj_normalized, adj, features, features_orig, size_factors, placeholders, is_training):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['features_orig']: features_orig})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    feed_dict.update({placeholders['size_factors']: size_factors})
    feed_dict.update({placeholders['is_training']: is_training})
    return feed_dict

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def mask_test_edges(adj):
    adj_row = adj.nonzero()[0]
    adj_col = adj.nonzero()[1]
    edges = []
    edges_dic = {}
    for i in range(len(adj_row)):
        edges.append([adj_row[i], adj_col[i]])
        edges_dic[(adj_row[i], adj_col[i])] = 1
    false_edges_dic = {}
    num_test = int(np.floor(len(edges) / 10.))
    num_val = int(np.floor(len(edges) / 20.))
    all_edge_idx = np.arange(len(edges))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    edges = np.array(edges)
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    test_edges_false = []
    val_edges_false = []
    while len(test_edges_false) < num_test or len(val_edges_false) < num_val:
        i = np.random.randint(0, adj.shape[0])
        j = np.random.randint(0, adj.shape[0])
        if (i, j) in edges_dic:
            continue
        if (j, i) in edges_dic:
            continue
        if (i, j) in false_edges_dic:
            continue
        if (j, i) in false_edges_dic:
            continue
        else:
            false_edges_dic[(i, j)] = 1
            false_edges_dic[(j, i)] = 1
        if np.random.random_sample() > 0.333 :
            if len(test_edges_false) < num_test :
                test_edges_false.append((i, j))
            else:
                if len(val_edges_false) < num_val :
                    val_edges_false.append([i, j])
        else:
            if len(val_edges_false) < num_val :
                val_edges_false.append([i, j])
            else:
                if len(test_edges_false) < num_test :
                    test_edges_false.append([i, j])
    
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def mask_test_express(adata):
    features = adata.X
    features_orig = adata.raw.X if FLAGS.use_raw_as_output else adata.X
    size_factors = adata.obs.size_factors

    row, col = np.nonzero(features)
    data = []
    for i in range(len(row)):
        data.append(features[row[i]][col[i]])
    data = np.array(data)
    sp_features = sp.csc_matrix((data, (row, col)), shape=(features.shape))
    
    num_test = int(np.floor(len(row) / 10.))
    num_val = int(np.floor(len(row) / 90.))
    all_features_idx = np.arange(len(data))
    np.random.shuffle(all_features_idx)
    val_data_idx = all_features_idx[:num_val]
    test_data_idx = all_features_idx[num_val:(num_val+num_test)]

    val_features = data[val_data_idx]
    test_features = data[test_data_idx]
    val_features_idx = []
    for val_i in val_data_idx:
        val_row = row[val_i]
        val_col = col[val_i]
        val_features_idx.append([val_row, val_col])
    test_features_idx = []
    for test_i in test_data_idx:
        test_row = row[test_i]
        test_col = col[test_i]
        test_features_idx.append([test_row, test_col])

    features = features.T
    return features, features_orig, size_factors, val_features, val_features_idx, test_features, test_features_idx
