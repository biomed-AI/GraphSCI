
import time
import os
import datetime
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import scanpy as sc
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_squared_error

from optimizer import  OptimizerSCI
from input_data import load_data
from network import GraphSCI
from preprocessing import *

os.environ['CUDA_VISIBLE_DEVICES'] = "4"

import warnings
warnings.filterwarnings('ignore')

now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d %H:%M:%S")

np.random.seed(42)
tf.set_random_seed(42)
os.environ['PYTHONHASHSEED'] = '0'

flags = tf.flags
FLAGS = flags.FLAGS
# args for training
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 50, 'Number of batch size for training.')
flags.DEFINE_float('weight_decay', 0.01, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')

# args for path
flags.DEFINE_string('output', '../output/', 'The direction for output files')

# args for single-cell datasets
flags.DEFINE_string('adata', '../data/splatter_data/counts_simulated_dataset1_1500x2500_dropout0.17.h5ad', 
                             'input file for adata.')
flags.DEFINE_string('adj', '../data/splatter_data/adj/true_counts_dataset1_1500x2500_dropout0.17_adj.npz', 'input adjacency.')

# args for data-preprocessing
flags.DEFINE_boolean('normalize_per_cell', True, 'If true, library size normalization is performed using \
                                                  the `sc.pp.normalize_per_cell` function in Scanpy and saved into adata \
                                                  object.')
flags.DEFINE_boolean('scale', True, 'If true, the input of the autoencoder is centered using \
                                    `sc.pp.scale` function of Scanpy.')                                                  
flags.DEFINE_boolean('log1p', True, 'If true, the input of the autoencoder is log transformed with a \
                                     pseudocount of one using `sc.pp.log1p` function of Scanpy.')
flags.DEFINE_boolean('use_raw_as_output', True, 'If true, the ground-truth of express data is adata.raw.X')


# make dirs
if FLAGS.output is not None:
    os.makedirs(FLAGS.output, exist_ok=True)
output_dir = os.path.join(FLAGS.output, now)

model_path = os.path.join(output_dir, 'checkpoint')
prediction_path = os.path.join(output_dir, 'prediction')
log_path = os.path.join(output_dir, 'log')

create_dir_if_not_exists(model_path)
create_dir_if_not_exists(prediction_path)
create_dir_if_not_exists(log_path)


adj, adata = load_data()

adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()


adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
features, features_orig, size_factors, val_features, val_features_idx, test_features, test_features_idx = mask_test_express(adata)

adj = adj_train
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'features_orig': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'size_factors': tf.placeholder(tf.float32),
    'is_training': tf.placeholder_with_default(True, shape=())
}

num_features = features.shape[1]
num_nodes = features.shape[0]

model = GraphSCI(placeholders, num_features, num_nodes)

pos_weight_adj = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm_adj = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


global_step = tf.Variable(0, trainable=False)
# Optimizer
with tf.name_scope('optimizer'):
    opt = OptimizerSCI(preds=(tf.reshape(model.z_adj, [-1]), tf.reshape(model.z_express, [-1])),
                       labels=(tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                               tf.reshape(placeholders['features_orig'], [-1])),
                       model=model,
                       num_nodes=num_nodes,
                       num_features=num_features,
                       pos_weight_adj=pos_weight_adj,
                       norm_adj=norm_adj,
                       global_step=global_step)

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

# Initialize session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver = tf.train.Saver(var_list=tf.global_variables())
sess.run(tf.global_variables_initializer())

def get_roc_score(edges_pos, edges_neg):

    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1 + np.exp(-x))
    
    # Predict on test set of edges
    feed_dict.update({placeholders['is_training']: False})
    adj_rec = sess.run(model.z_adj, feed_dict=feed_dict).reshape([num_nodes, num_nodes])
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])
    
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_mse_score(features, features_idx, epoch):

    # Predict on test set of features
    feed_dict.update({placeholders['is_training']: False})
    features_rec = sess.run(model.z_express, feed_dict=feed_dict).reshape([num_features, num_nodes])

    adata_copy = adata.copy()
    adata_copy.X = features_rec
    output_file = os.path.join(prediction_path, 'graphsci_tf_simulated_counts_%d.h5ad' % epoch)
    adata_copy.write(output_file)

    preds = []
    for idx in features_idx:
        preds.append(features_rec[idx[0], idx[1]])
    
    mse_score = mean_squared_error(np.array(features), np.array(preds))

    return mse_score


# Train model
log_file = os.path.join(log_path, 'log.txt')
fp = open(log_file, 'w')
learning_rates = []
costs = []
zinb_losses = []
adj_losses = []
for epoch in range(FLAGS.epochs):

    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, features_orig, size_factors, placeholders, is_training=True)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({global_step: epoch})
    
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.cost_adj, opt.cost_express, opt.kl_express, opt.learning_rate], feed_dict=feed_dict)

    # Compute average loss
    cost = outs[1]
    cost_adj = outs[2]
    cout_express = outs[3]
    kl = outs[4]
    learning_rates.append(outs[5])
    costs.append(cost)
    zinb_losses.append(cout_express)
    adj_losses.append(cost_adj)
    mse_score = get_mse_score(val_features, val_features_idx, epoch+1)
    roc_score, ap_score = get_roc_score(val_edges, val_edges_false)

    log_str = "[Epoch%d] train_loss %.6f adj_loss %.6f express_loss %.6f mse_score %.6f roc_score %.6f kl %.6f" \
                % (epoch + 1, cost, cost_adj, cout_express, mse_score, roc_score, kl)
    make_log(log_str, fp)



fp.close()
print("Optimization Finished!")
# print(learning_rates)

lr_file = open('lr_hidden16.txt', 'w')
lr_file.write(str(learning_rates))
lr_file.close()

adj_loss_file = open('adj_loss_hidden16.txt', 'w')
adj_loss_file.write(str(adj_losses))
adj_loss_file.close()

zinb_loss_file = open('zinb_loss_hidden16.txt', 'w')
zinb_loss_file.write(str(zinb_losses))
zinb_loss_file.close()

loss_file = open('loss_hidden16.txt', 'w')
loss_file.write(str(costs))
loss_file.close()

# test-set
mse_score_test = get_mse_score(test_features, test_features_idx, 0)
# roc_score_test, ap_score_test = get_roc_score(test_edges, test_edges_false)
feed_dict.update({placeholders['is_training']: False})
rec = sess.run(model.z_express, feed_dict=feed_dict).reshape([num_features, num_nodes])

adata_copy = adata.copy()
adata_copy.X = rec

# warning! this may overwrite adata.X
output_file = os.path.join(prediction_path, 'graphsci_tf_simulated_counts.h5ad')
adata_copy.write(output_file)

model_file = os.path.join(model_path, 'graphsci_tf_model.ckpt')
saver.save(sess, model_file)

sc.pp.normalize_per_cell(adata_copy)
sc.pp.log1p(adata_copy)
sc.pp.pca(adata_copy)

# sil_score = silhouette_score(adata_copy.obsm.X_pca[:, :2], adata_copy.obs.Group)
# print(sil_score)
# print(rec)

# print("Begin calculate PCA")
# from sklearn.decomposition import KernelPCA
# from sklearn.preprocessing import LabelEncoder
# X, y = np.array(adata_copy.X), np.array(adata_copy.obs.Group)
# lf=LabelEncoder().fit(y)
# y_true=lf.transform(y)
# transformer = KernelPCA(n_components=10, kernel='rbf')
# X_transformed = transformer.fit_transform(X)
# sil_score2 = silhouette_score(X_transformed[:, :2], adata_copy.obs.Group)
# print(sil_score2)

# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=3, n_init=20)
# y_pred = kmeans.fit(X_transformed).labels_

# acc = np.round(cluster_acc(y_true, y_pred), 5)
# nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
# ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
# print('KernelPCA + KMeans result: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))