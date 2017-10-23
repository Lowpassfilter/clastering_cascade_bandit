from useful_fns import *
import numpy as np
from numpy.linalg import inv
import time

def CascadeLinUCB(timestep, K, sub_X, sub_bool, synthetic, group_size, group_theta, opt_list, w_list):
    d = sub_X.shape[1]
    E = sub_bool.shape[0]
    user_size = sub_bool.shape[1]
    c = constant_c(K, d, timestep)

    M = np.matrix(constant_lambda(K)*np.identity(d))
    B = np.matrix(np.zeros(d)).T
    
    regrets_list=[]
    regret=0.0

    y = probability_by_count(sub_bool)

    opt, w = real_optimal(sub_bool, K)
    start = time.time()
    for t in range(timestep):
        user_id=np.random.randint(user_size)
        if synthetic:
            group_id = user_id/group_size
            opt = opt_list[group_id]
            w = w_list[group_id]
            theta_stars = group_theta[:, group_id]

        M_inv=inv(M)
        theta=np.matmul(M_inv, B)
        U=np.zeros(E)
        
        for e in range(E):
            x_e=np.matrix(sub_X[e,:]).T
            temp1=np.matmul(x_e.T, theta)
            temp2=math.sqrt(np.matmul(np.matmul(x_e.T, M_inv), x_e))
            U[e]=min(temp1 + np.random.uniform(0, 1) * c * temp2, 1)
            
        A=np.argsort(U)[E-1:E-1-K:-1]

        if synthetic:
            feedback=st.bernouli_array(w)
        else:
            feedback=sub_bool[:,user_id].A1

        reward, Ct=observe_ucb(A, feedback)
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
            print regret, str(t/1000)+'k'+" step \t"+str(duration)+" s \t"+"UCB"

        M, B = update_statistics(M,B,Ct,K, sub_X, A, feedback)
    print regret
    return regrets_list