import scipy.io as sio
import scipy.sparse as spp
import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize




def extract_rows(top_k, sparse_matrix):
    business_review_count=sparse_matrix.getnnz(axis=1)
    business_count=business_review_count.shape[0]
    top_k_index = np.argsort(business_review_count)[business_count-1: business_count -1 -top_k: -1]
    # top_k_index = np.random.choice(business_count, top_k, replace=False)
    matrix=spp.vstack([sparse_matrix.getrow(i) for i in top_k_index])
    return matrix

def extract_cols(top_k, sparse_matrix):
    user_review_count=sparse_matrix.getnnz(axis=0)
    user_count=user_review_count.shape[0]

    top_k_index=np.argsort(user_review_count)[user_count-1: user_count-1-top_k:-1]
    # top_k_index=np.random.choice(user_count, top_k, replace=False)
    matrix=spp.hstack([sparse_matrix.getcol(i) for i in top_k_index])
    return matrix





def load_sparse_matrix(file_name):
    data_list = []
    row_indics_list = []
    col_indics_list = []
    
    row_dict = {}
    col_dict = {}
    new_row_index = 0
    new_col_index = 0
    
    rf = open(file_name)
    
    l = rf.readline()
    count = 0
    for line in rf:
        pairs = line.strip('\n').split(',')
        row_index = 0
        col_index = 0
        if row_dict.has_key(int(pairs[1])):
            row_index = row_dict[int(pairs[1])]
        else:
            row_dict[int(pairs[1])] = new_row_index
            row_index = new_row_index
            new_row_index += 1

        if col_dict.has_key(int(pairs[0])):
            col_index = col_dict[int(pairs[0])]
        else:
            col_dict[int(pairs[0])] = new_col_index
            col_index = new_col_index
            new_col_index += 1
        
        data_list.append(float(pairs[2]))
        row_indics_list.append(row_index)
        col_indics_list.append(col_index)


    data = np.array(data_list)
    rows = np.array(row_indics_list)
    cols = np.array(col_indics_list)
    
    s_m = spp.csr_matrix((data, (rows, cols)))

    return s_m


def get_reduced_matrix(k_business, k_user):
    s_m = load_sparse_matrix("ml-20m/ratings.csv")
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
    new_matrix = np.matrix(new_matrix)
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



reduced_matrix = get_reduced_matrix(1000, 1000)
new_matrix = split_matrix(reduced_matrix)
full_matrix = convert_bool_matrix(new_matrix.toarray())
np.savetxt("ml-20m/movie_best_pool.csv",full_matrix,  fmt='%d', delimiter=",")