import argparse
import numpy as np 
from kneed import KneeLocator
import matplotlib.pyplot as plt

import jpype
import jpype.imports
from jpype.types import *
print(jpype.getDefaultJVMPath())
jpype.startJVM("-Xmx10000m", classpath=['./optimaltransport.jar'])
from optimaltransport import Mapping

def e_dist(A, B):
    A_n = (A**2).sum(axis=1).reshape(-1,1)
    B_n = (B**2).sum(axis=1).reshape(1,-1)
    inner = np.matmul(A, B.T)
    return A_n - 2*inner + B_n

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


############# The main function #################

def GTOT(x, init_mu=-1, delta=0.01, S=1, seed=1):
    n, d = x.shape
    mu = init_mu * np.ones(d)
    supplies = [1.0]*n
    demands = [1.0]*n

    np.random.seed(seed+1)
    z1 =  np.random.normal(0, 1, (n, d))
    z = z1 + mu 
    cost_vector = e_dist(x, z)
    cost_vector /= cost_vector.max()
    ##
    gtSolver = Mapping(n, supplies, demands, cost_vector, delta)
    APinfo = np.array(gtSolver.getAPinfo())
    F = np.array(gtSolver.getFlow())
    # clean_mask = (APinfo[:,2] >= 1000)
    # APinfo_cleaned = APinfo[clean_mask]
    totalFlow = sum(APinfo[:, 2])
    cumFlow = np.cumsum((APinfo[:,2]).astype(int))
    flowProgress = (cumFlow)/(1.0 * totalFlow)

    phase_dual = np.stack((flowProgress, 1.0 * APinfo[:,0], np.cumsum(1.0 * APinfo[:, 4]),\
                                1.0 * APinfo[:, 4], 1.0 * APinfo[:, 6]), axis=-1)

    AP_num = np.unique(phase_dual[:, 1], return_counts=True)[1]
    index_first = np.unique(phase_dual[:, 1], return_index=True)[1]
    index_last = index_first + AP_num - 1
    phase_dual_unique = phase_dual[index_last, :]
    phase_dual_unique = np.concatenate((phase_dual_unique, np.expand_dims(AP_num, axis=-1)), axis=-1)

    smooth_by_flowrate = 0.01
    move_step = n*smooth_by_flowrate
    xpr = running_mean(flowProgress, int(move_step))
    ypr = running_mean(APinfo[:,4], int(move_step))
    
    ignore_transport = np.where(xpr > 0.3)
    xpr = xpr[ignore_transport]
    ypr = ypr[ignore_transport]

    kneedle = KneeLocator(ypr, xpr, S, curve="concave", direction="increasing")
    k_cut = kneedle.knee_y # kneedle method

    good_set = np.zeros(n, dtype=bool)
    if k_cut > 0.5:
        good_set[0:(k_cut*n).astype(int)] = True
    else:
        good_set[0:int(n/2)] = True

    mu = np.mean(x[good_set, :], axis=0)
    mu_out = np.mean(x[~good_set, :], axis=0)

    return mu, mu_out, k_cut, xpr, ypr

############ An application ################

parser = argparse.ArgumentParser()
parser.add_argument('--nexp', type=int, default=10)
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--delta', type=float, default=0.01)
parser.add_argument('--eps', type=float, default=0.2)
parser.add_argument('--mu_true', type=float, default=0)
parser.add_argument('--mu_cont', type=float, default=2)
parser.add_argument('--mu_init', type=float, default=0)
args = parser.parse_args()
print(args)

NUM_EXPERIMENTS = int(args.nexp)
n = int(args.n)
delta = args.delta
eps = args.eps
mu_true = args.mu_true
mu_cont = args.mu_cont
mu_init = args.mu_init

d = 2
sigma_true = np.eye(d)
contprop = np.int(n * eps)
mu_in = []
mu_out = []
mu_in_est_error = []
mu_out_est_error = []
k_cut = []

for i in range(NUM_EXPERIMENTS):
    ### generate the contaminated dataset
    np.random.seed(i)  
    x = np.zeros((n, d)) 
    x[0:(n - contprop), :] = np.random.normal(mu_true, 1, ((n - contprop), d))
    x[(n - contprop):, :] = np.random.normal(mu_cont, 1, (contprop, d))

    ### Estimate mean using ROBOT 
    mu_in_hat, mu_out_hat, k, x, y = GTOT(x, mu_init, delta, S=1, seed=i)
    mu_in.append(mu_in_hat)
    mu_out.append(mu_out_hat)
    mu_in_est_error.append(np.linalg.norm(mu_in_hat-mu_true))
    mu_out_est_error.append(np.linalg.norm(mu_out_hat-mu_cont))
    k_cut.append(k)

    plt.plot(x, y)
    plt.axvline(x = k, ls='--', c='black', label='knee')
    plt.xlabel('flow progress', fontsize=14)
    plt.ylabel('1st derivative', fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig("syn_N{}_mu_count{}_eps{}_no{}.png".format(n, int(mu_cont), eps, i+1))
    plt.clf()

print("k_cut {}({})".format(np.round(np.mean(k_cut),3), np.round(np.std(k_cut),3)))
print("OT profile inlier mean estimation {}({})".format(np.round(np.mean(mu_in),3), np.round(np.std(mu_in),3)))
print("OT profile achieved inlier estimation err {}({}) ".format(np.round(np.mean(mu_in_est_error),3), np.round(np.std(mu_in_est_error),3)))
print("OT profile outlier mean estimation {}({})".format(np.round(np.mean(mu_out),3), np.round(np.std(mu_out),3)))
print("OT profile achieved outlier estimation err {}({}) ".format(np.round(np.mean(mu_out_est_error),3), np.round(np.std(mu_out_est_error),3)))