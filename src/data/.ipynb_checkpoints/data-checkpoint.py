import numpy as np
import pandas as pd
import scipy.stats as ss
import math


def gen_data(J = 1000, q = 0.9):
    """
    J = maximum number of hypothesis tests
    N = maximum number of samples per test
    random_q = whether prob. of null (q) should be random for each test
    """
    mu = (0,2)
    (q0,alt_true,z) = ([],[],[])
    
    for j in range(J):
        p_j = 1 - q
        alt_true_j = ss.bernoulli.rvs(p_j)
        z_j = ss.norm.rvs(loc=mu[alt_true_j], size=1)  # maximum number of samples available for testing
        q0.append(1-p_j)
        alt_true.append(alt_true_j)
        z.append(z_j)
    return (z, np.array(alt_true), q0)

def microarray_data():
    prostate = pd.read_csv('prostate.csv')
    #prostate = prostate.iloc[np.random.permutation(len(prostate))]
    prostate_normal = prostate[prostate.columns[np.arange(50)]]
    prostate_tumor = prostate[prostate.columns[np.arange(50, 102)]]
    normal_mean = prostate_normal.apply(np.mean, axis = 1)
    transform_tumor = pd.DataFrame(np.zeros([6033, 52]))
    for i in np.arange(len(prostate_tumor)):
        transform_tumor.iloc[[i]] = prostate_tumor.iloc[[i]] - normal_mean[i]
    normal_sd = prostate_normal.std(axis = 1)
    transform_tumor *= (1/normal_sd)
    transform_mean = transform_tumor[np.arange(2)].apply(np.mean, axis = 1)
    overall_tmean = np.mean(transform_mean)
    q0 = 1-1/(1+np.exp(-2*(transform_mean - math.log10(4) * (1/normal_sd)))) #transform mean only first two
    z = transform_tumor[np.arange(2,52)] #50 samples (col 3-52)
    
    return (z, normal_sd, q0)



def gen_data_batch(J = 1000, K=10, N=1000, a = 1,b = 1):
    """
    J = maximum number of batches
    K = number of hypothesis per batch
    N = maximum number of samples per hypothesis
    """
    mu = (0,2)
    z, alt_true, q0 = np.zeros((J, K, N)), np.zeros((J, K)), np.zeros((J, K))
    
    for j in range(J):
        for k in range(K):
            p_j = ss.beta.rvs(a,b)
            alt_true[j][k] = ss.bernoulli.rvs(p_j)
            z[j][k][:] = ss.norm.rvs(loc=mu[int(alt_true[j][k])], size=N)  # maximum number of samples available for testing       
            q0[j][k] = 1-p_j

    return (z, alt_true, q0)
