import scipy.io as sio
import scipy.sparse as spp
import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize



def reduce_by_item(full_pool, top_k):
    item_review = full_pool.sum(axis=1)
    total_item =full_pool.shape[0]
    if top_k<total_item:
        top_k_index = np.argsort(item_review)[total_item - 1: total_item - 1 - top_k: -1]
    else:
        top_k_index = np.argsort(item_review)[::-1]
    new_array = full_pool[top_k_index,:]
    return new_array


def add_bias_col(X):
    bias=np.ones((X.shape[0], 1))
    X=np.concatenate((bias, X), axis=1)
    return X

def get_data(item, hist_user, criterion_user, feature):
    f=open("ml-20m/movie_random_pool.csv")
    full_pool=np.loadtxt(f, delimiter=',')
    
    item_select_pool=reduce_by_item(full_pool, item)
    train_matrix = item_select_pool[:, [i for i in range(hist_user)]]
    
    test_matrix = item_select_pool[:, np.random.choice([i for i in range(hist_user, item_select_pool.shape[1])], criterion_user, replace=False)]
    
    u, s, vt=np.linalg.svd(train_matrix)

    sigma= np.diag(s[0:feature])
    u=u[:, [i for i in range(feature)]]

    X=normalize(u, axis=1, norm='l2')

    X=add_bias_col(X)

    return np.matrix(X), np.matrix(test_matrix)

