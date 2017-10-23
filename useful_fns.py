import scipy as sp
import numpy as np
import math
from numpy.linalg import inv
import sys
import stimulation as st
import clustering as cl
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import json
import time
import datetime

def constant_alpha(K, d, n):
    return 8 * math.sqrt(d)

def constant_beta(K, d, n, plambda):
    temp=d*math.log(1+float(n)/(d*plambda)) + 2*math.log(4 * n * K)
    return math.sqrt(temp) + math.sqrt(plambda)

def constant_lambda(K):
    return K

def constant_c(K, d, n):
    temp=d*math.log(1+float(n)/d) + 2*math.log(n * K)
    return math.sqrt(temp) + math.sqrt(K)

def probability_by_count(bool_matrix):
    m=bool_matrix.shape[1]
    w=np.sum(bool_matrix, axis=1)/m
    return np.matrix(w).T

def real_optimal(bool_matrix, K):
    e=bool_matrix.shape[0]
    m=bool_matrix.shape[1]
    w=np.sum(bool_matrix, axis=1)/m
    w=w.A1
    return reward_function(w, K), w

def stimulated_optimal(sub_X, theta_stars, K):
    w=np.matmul(sub_X, theta_stars).A1
    return reward_function(w, K), w

def group_stimulated_optimal(sub_X, theta_groups, K):
    w_list = []
    opt_list = []
    for i in range(theta_groups.shape[1]):
        opt, w = stimulated_optimal(sub_X, theta_groups[:, i], K)
        w_list.append(w)
        opt_list.append(opt)
    return opt_list, w_list

def reward_function(w, K):
    e=w.shape[0]
    if K < e:
        best_k=np.sort(w)[e-1:e-1-K:-1]
    else:
        best_k=np.sort(w)[::-1]
    opt=1
    for i in range(K):
        opt=opt*(1-best_k[i])
    return 1-opt

def observe_ucb(A, feedback):
    for i in range(A.shape[0]):
        if feedback[A[i]] == 1:
            return 1, i
    return 0, sys.maxint


def observe_dcm(A, feedback):
    Ct= sys.maxint
    r = 0
    for i in range(A.shape[0]):
        if feedback[A[i]] == 1:
            Ct=i
            r=1
    return r, Ct

def update_statistics(M,B,Ct,K, X, A, feedback):
    for i in range(min(Ct + 1, K)):
        e=A[i]
        x_e=np.matrix(X[e,:]).T
        M = M + np.matmul(x_e, x_e.T)
        B = B + feedback[e]*x_e
    return M, B


def select_sub(X, bool_matrix, E, user_size):
    total_item = bool_matrix.shape[0]
    total_user = bool_matrix.shape[1]
    row_ids = np.random.choice(total_item, E, replace = False)
    sub_X = X[row_ids,:]
    temp_bool = bool_matrix[row_ids,:]
    col_ids = np.random.choice(total_user, user_size, replace = False)
    # col_ids = user_id_of_different_group(temp_bool, user_size)
    sub_bool = temp_bool[:, col_ids]
    return sub_X, sub_bool

def user_id_of_different_group(bool_matrix, user_size):
    cluster_number = 2
    group_size = user_size/cluster_number
    group_size_list = [group_size for i in range(cluster_number - 1)]
    group_size_list.append(user_size - group_size*(cluster_number -1))
    group = KMeans(n_clusters = cluster_number).fit(bool_matrix.T)

    id_list = []
    for i in range(cluster_number):
        id_list = id_list + np.random.choice(np.where(group.labels_ == i)[0], group_size_list[i], replace = False).tolist()
    return id_list
    




def user_distance_matrix(bool_matrix):
    norm_matrix = np.matrix(np.zeros(bool_matrix.shape))
    user_count = bool_matrix.shape[1]
    for i in range(user_count):
        norm_matrix[:, i] = normalize(bool_matrix[:, i], axis=0, norm='l2')

    distance_matrix = [[np.linalg.norm( norm_matrix[:, i] - norm_matrix[:, j] ) for i in range(user_count)] for j in range(user_count)]
    distance_matrix = np.matrix(distance_matrix)

    return distance_matrix

def real_ctr(w, A):
    e=A.shape[0]
    opt=1
    for i in range(e):
        opt=opt*(1-w[A[i]])
    opt=1-opt
    return opt


#### those functions below has nothing to do with the algorithms, they provide peripheral help

    

def plot_curves(dict_list):    
    for d in dict_list:
        X = np.arange(len(d['data']))
        plt.plot(X, d['data'], color = d['color'], linewidth= d['linewidth'], linestyle = d['linestyle'], label = d['label'])
    plt.legend(loc='upper left', frameon=False)
    plt.show()

def load_result(file_path):
    rf = open(file_path)
    para_dict = json.loads(rf.readline())
    dict_list=[]
    for line in rf:
        dict_list.append(json.loads(line))
    return para_dict, dict_list

def smooth(data, window):
    prefix = [sum(data[0:i])/i for i in range(1, window)]
    new_data = prefix + [sum(data[i:i+window])/window for i in range(len(data) - window)]
    return new_data

def save_data(para_dict, dict_list, name=""):
    ts = time.time()
    # name = raw_input("Please enter something: ")
    file_path = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')+"-"+name+".txt"
    json_file = open(file_path, "wb")
    json.dump(para_dict, json_file)
    for d in dict_list:
        json_file.write('\r\n')
        json.dump(d, json_file)
    json_file.close()