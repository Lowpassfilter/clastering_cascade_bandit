import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import ortho_group



def bernouli_array(p_array):
    prob=[np.random.binomial(1, p_array[i]) for i in range(p_array.shape[0])]
    return prob

def generate_theta(width):
    a=np.matrix(np.random.uniform(size=width)).T
    a=normalize(a, axis=0, norm='l2')
    unit=np.matrix(1)
    a=np.concatenate((unit, a), axis=0)
    a= a/2
    return a

def similiarity(theta1, theta2):
    temp=theta1.T
    norm1=np.linalg.norm(theta1, axis=0)[0]
    if norm1 == 0:
        norm1 += 0.00001
    norm2=np.linalg.norm(theta2, axis=0)[0]
    if norm2 == 0:
        norm2 += 0.00001
    product=np.matmul(temp, theta2)[0,0]

    return product/(norm1*norm2)


def generate_group_theta(width, group_number, threshold):
    m = generate_theta(width)
    while m.shape[1] < group_number:
        v = generate_theta(width)
        if valid_theta(m, v, threshold):
            m =  np.concatenate((m, v), axis = 1)
    return m

def orthogonal_theta(width, group_number):
    if group_number >= width:
        raise ValueError("more group_number than feature number")
    X = ortho_group.rvs(width)
    X = X[:, [i for i in range(group_number)]]
    head = np.matrix(np.ones(group_number))
    X = np.concatenate((head, X), axis = 0)
    return X/2
    

def valid_theta(m, v, theshold):
    for i in range(m.shape[1]):
        if np.linalg.norm(m[:, i] - v) < theshold:
            return False
    return True
    
def feature_vector(item, d):
    w = np.random.normal(size=(item, d))
    w = normalize(w, axis=1, norm='l2')
    unit = np.matrix(np.ones(item)).T
    w = np.concatenate((unit, w), axis=1)

    return w
