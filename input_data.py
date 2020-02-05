
import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow as tf
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

flags = tf.flags
FLAGS = flags.FLAGS

def load_data():
    adata = read_dataset(FLAGS.adata,
                         transpose=False,
                         test_split=False,
                         copy=False)
    
    # check for zero genes
    nonzero_genes, _ = sc.pp.filter_genes(adata.X, min_counts=1)
    assert nonzero_genes.all(), 'Please remove all-zero genes before using DCA.'

    adata = normalize(adata,
                      filter_min_counts=False, # no filtering, keep cell and gene idxs same
                      size_factors=FLAGS.normalize_per_cell,
                      normalize_input=FLAGS.scale,
                      logtrans_input=FLAGS.log1p)
    
    adj = load_adj(FLAGS.adj)

    return adj, adata

def load_adj(adj):
    if isinstance(adj, str):
        adj = sp.load_npz(adj)
    elif sp.issparse(adj):
        adj = adj
    else:
        raise NotImplementedError

    return adj

def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def read_dataset(adata, transpose=False, check_normalization=False, test_split=False, copy=False):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        if adata[-3:] == 'csv':
            counts = pd.read_csv(adata.T.values, index_col=0)
            adata = sc.AnnData(counts.values)
            adata.obs_names = list(counts.columns)
            adata.var_names = list(counts.index)
        else:
            adata = sc.read(adata, first_column_names=True)
    else:
        raise NotImplementedError

    if check_normalization:
        # check if observations are unnormalized using first 10
        X_subset = adata.X[:10]
        norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
        if sp.issparse(X_subset):
            assert (X_subset.astype(int) != X_subset).nnz == 0, norm_error
        else:
            assert np.all(X_subset.astype(int) == X_subset), norm_error

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['dca_split'] = spl.values
    else:
        adata.obs['dca_split'] = 'train'

    adata.obs['dca_split'] = adata.obs['dca_split'].astype('category')
    print('Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata
