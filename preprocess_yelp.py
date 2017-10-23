import scipy.io as sio
import scipy.sparse as spp
import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize
from manifest import *



def extract_rows(top_k, sparse_matrix):
    business_review_count=sparse_matrix.getnnz(axis=1)
    business_count=business_review_count.shape[0]
    top_k_index = np.argsort(business_review_count)[business_count-1: business_count -1 -top_k: -1]
    matrix=spp.vstack([sparse_matrix.getrow(i) for i in top_k_index])
    return matrix

def extract_cols(top_k, sparse_matrix):
    user_review_count=sparse_matrix.getnnz(axis=0)
    user_count=user_review_count.shape[0]

    top_k_index=np.argsort(user_review_count)[user_count-1: user_count-1-top_k:-1]
    matrix=spp.hstack([sparse_matrix.getcol(i) for i in top_k_index])
    return matrix


def load_sparse_matrix(s_file_name, business_rowindex_dict_file_name, user_colindex_dict_file_name):
    s_m=sio.mmread(s_file_name)
    business_rowindex_dict=np.load(business_rowindex_dict_file_name).item()
    user_colindex_dict=np.load(user_colindex_dict_file_name).item()
    return s_m, business_rowindex_dict, user_colindex_dict




def get_reduced_matrix(k_business, k_user):
    s_m, business_rowindex_dict, user_colindex_dict=load_sparse_matrix(FULL_MATRIX_FILE_NAME, FULL_BUSINESS_ROWINDEX_DICT_FILE_NAME, FULL_USER_COLINDEX_DICT_FILE_NAME)
    row_reduced_matrix=extract_rows(k_business*3, s_m)
    reduced_matrix=extract_cols(k_user, row_reduced_matrix)
    reduced_matrix = extract_rows(k_business, reduced_matrix)
    return reduced_matrix

def split_matrix(reduced_matrix):
    col_seeds=np.random.permutation(reduced_matrix.shape[1])
    row_seeds=np.random.permutation(reduced_matrix.shape[0])

    temp_matrix=spp.hstack([reduced_matrix.getcol(col_seeds[i])  for i in range(col_seeds.shape[0])])
    new_matrix=spp.vstack([temp_matrix.getrow(row_seeds[i]) for i in range(row_seeds.shape[0])])

    return new_matrix


def add_bias_col(X):
    bias=np.ones((X.shape[0], 1))
    X=np.concatenate((bias, X), axis=1)
    return X

def convert_bool_matrix(test_matrix):
    x,y=sp.where(test_matrix>0)
    new_matrix=np.zeros(test_matrix.shape)
    new_matrix[x,y]=1
    return new_matrix

def get_data(item, user, feature):
    reduced_matrix = get_reduced_matrix(item,user)
    train_matrix, test_matrix, perm_seeds=split_matrix(reduced_matrix)
    bool_train=convert_bool_matrix(train_matrix.toarray())
    u, s, vt=np.linalg.svd(bool_train)

    sigma= np.diag(s[0:feature])
    u=u[:, [i for i in range(feature)]]

    # X=np.matmul(u, sigma)
    X=normalize(u, axis=1, norm='l2')

    X=add_bias_col(X)
    return X, convert_bool_matrix(test_matrix.toarray()), perm_seeds, s


# reduced_matrix = get_reduced_matrix(1000, 1000)
# new_matrix = split_matrix(reduced_matrix)
# full_matrix = convert_bool_matrix(new_matrix.toarray())
# np.savetxt("yelp_data/yelp_pool.csv",full_matrix,  fmt='%d', delimiter=",")


f=open("yelp_data/yelp_pool.csv")
a=np.loadtxt(f, delimiter=',')
print a.shape
