from useful_fns import *
import numpy as np
from numpy.linalg import inv
import time

def ClusteringCascadeLinUCB(timestep, K, sub_X, sub_bool, synthetic, edge_mode, cut_mode, group_size, group_theta, opt_list, w_list):
    d = sub_X.shape[1]
    E = sub_bool.shape[0]
    user_size = sub_bool.shape[1]
    alpha = constant_alpha(K, d, timestep)
    plambda = constant_lambda(K)
    beta = constant_beta(K, d, timestep, plambda)
    triger_counts = np.zeros(user_size)

    regrets_list = []
    regret = 0.0

    M_set = np.zeros((user_size, d, d))
    B_set = np.zeros((user_size, d))
    theta_set = np.zeros((user_size, d))

    distance_matrix = np.ones((user_size, user_size))
    
    opt, w = real_optimal(sub_bool, K)
    G = cl.Component_Graph(user_size, distance_matrix, edge_mode)
    start = time.time()
    for t in range(timestep):
        user_id = np.random.randint(user_size)
        
        if synthetic:
            group_id = user_id/group_size
            opt = opt_list[group_id]
            w = w_list[group_id]
        
        neighbors = G.neighbor_list(user_id)
        M = np.identity(d)*plambda
        B = np.zeros(d)
        for i in neighbors:
            M += M_set[i,:,:]
            B += B_set[i,:]
        M_inv = np.matrix(inv(M))
        B = np.matrix(B).T
        theta_t = np.matmul(M_inv, B)
        
        U=np.zeros(E)
        for e in range(E):
            x_e = np.matrix(sub_X[e, :]).T
            temp1 = np.matmul(x_e.T, theta_t)
            temp2 = math.sqrt(np.matmul(np.matmul(x_e.T, M_inv), x_e))
            U[e] = min(temp1 + np.random.uniform(0, 1) * beta * temp2, 1)

        A = np.argsort(U)[E-1:E-1-K:-1]
        
        if synthetic:
            feedback = st.bernouli_array(w)
        else:
            feedback = sub_bool[:,user_id].A1

        reward, Ct = observe_ucb(A, feedback)
        ctr = real_ctr(w, A)
        
        if synthetic:
            regret +=opt - ctr
        else:
            regret += reward
            
        regrets_list.append(regret)

        if t % 1000 == 0:
            end = time.time()
            duration = int(end-start)
            start = end
            print regret, str(t/1000)+'k'+" step \t"+str(duration)+" s \t"+str(len(G.graphs))+" clusters"

        Kt = min(Ct, K-1)
        triger_counts[user_id] = triger_counts[user_id] + Kt


        M_user = np.matrix(M_set[user_id, :, :])
        B_user = np.matrix(B_set[user_id, :]).T
        M_user, B_user = update_statistics(M_user, B_user, Ct, K, sub_X, A, feedback)

        M_set[user_id, :, :] = M_user
        B_set[user_id, :] = B_user.A1
        theta_set[user_id, :] = np.matmul(inv(M_user + plambda * np.identity(d)), B_user).A1

        theta_user = np.matrix(theta_set[user_id, :]).T

        sum1 = math.sqrt(1 + math.log(1 + triger_counts[user_id])) / (1 + triger_counts[user_id])
        for i in neighbors:
            if not i == user_id:
                theta_i = np.matrix(theta_set[i, :]).T
                distance = np.linalg.norm(theta_user - theta_i)
                sum2 = math.sqrt(1 + math.log(1 + triger_counts[i])) / (1 + triger_counts[i])
                thresh_hold = alpha * (sum1 + sum2)
                if distance > thresh_hold:
                    G.remove_edges(user_id, i, cut_mode, distance)
                distance_matrix[i, user_id] = distance_matrix[user_id, i] = distance
        G.update_graph(user_id)   
    print regret
    return regrets_list