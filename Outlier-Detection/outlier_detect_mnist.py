import argparse
import numpy as np
import ot
import os
import sys
import time
import tensorflow as tf
import scipy
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

import jpype
import jpype.imports
from jpype.types import *
print(jpype.getDefaultJVMPath())
jpype.startJVM("-Xmx10000m", classpath=['./optimaltransport.jar'])
from optimaltransport import Mapping

from kneed import KneeLocator


"""
Relevant parts of the code have been adapted from :
https://github.com/debarghya-mukherjee/Robust-Optimal-Transport/blob/main/ROBOT_mnist_outlier_detection.py
"""

def moving_average(a, n=10) :
    kernel = np.ones(n) / (n * 1.0)
    a_c = np.convolve(a, kernel, mode='same')
    return a_c

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def c_dist(b, a):
    return cdist(a, b, metric='minkowski', p=1)

def e_dist(A, B):
    A_n = (A**2).sum(axis=1).reshape(-1,1)
    B_n = (B**2).sum(axis=1).reshape(1,-1)
    inner = np.matmul(A, B.T)
    return A_n - 2*inner + B_n

def data_prep(eps=0.2, n=1000):
    # eps = Contamination proportion
    # n = number of samples
    ############ Creating pure and contaminated mnist dataset ############

    if os.path.exists('./mnist.npy'):
        mnist = np.load('./mnist.npy') # 60k x 28 x 28
        mnist_labels = np.load('./mnist_labels.npy').ravel().astype(int) # 60k x 1 # originally UINT8
    else:
        (mnist, mnist_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        np.save('mnist.npy', mnist)
        np.save('mnist_labels.npy', mnist_labels)


    k = 1 - eps # rate of inlier (eps : outlier)

    total = np.arange(len(mnist_labels))
    inlier_indx = total[mnist_labels < 5]
    outlier_indx = total[mnist_labels > 4]

    inlier_indx_a = np.random.permutation(inlier_indx)[:n]
    inlier_indx_b1 = np.random.permutation(inlier_indx)[:int(n*k)]
    outlier_indx_b2 = np.random.permutation(outlier_indx)[:samples_no - int(n * k)]

    # print(inlier_indx_a.shape, inlier_indx_b1.shape, outlier_indx_b2.shape)

    # random case
    mix_indx = np.concatenate((outlier_indx_b2, inlier_indx_b1)) # 2k, 8k
    rand_mask = np.random.permutation(np.arange(len(mix_indx)))
    real_outlier_mask = (rand_mask <= n*(1-k))
    real_inlier_mask = (rand_mask > n*(1-k))
    mix_indx = mix_indx[rand_mask]


    a, a_labels  = mnist[inlier_indx_a, :, :], mnist_labels[inlier_indx_a]
    b, b_labels  = mnist[mix_indx, :, :], mnist_labels[mix_indx]

    # print(a_labels[:10])
    # print(b_labels[:10])
    a_idx, b_idx = (1-np.array(a_labels < 5).astype(int)), np.array(b_labels > 4).astype(int)

    # im2double
    a = a/255.0
    b = b/255.0

    a = a.reshape(-1, 784)
    b = b.reshape(-1, 784)

    # print(a.shape, b.shape)

    a = a / a.sum(axis=1, keepdims=1)
    b = b / b.sum(axis=1, keepdims=1)

    # print(a.shape, b.shape, np.sum(a[0]), np.sum(b[0]))

    # print("clean, dirty", a.shape, b.shape)

    return a, a_labels, a_idx, b, b_labels, b_idx


def ROBOT_lambda_selection(X=None, Y=None, iter=30, sample_batch=5000):
    ######### Selection of lambda
    # X is dirty Y is clean
    lambda_vec = np.zeros(iter)
    for j in range(iter):
        Y1 = Y[np.random.choice(Y.shape[0], sample_batch, replace=False)]
        Y2 = Y[np.random.choice(Y.shape[0], sample_batch, replace=False)]

        cost_now = e_dist(Y1, Y2)

        Pival, Pi = ot.emd2(np.ones(sample_batch)/sample_batch, np.ones(sample_batch)/sample_batch, cost_now, processes=4, numItermax=1e6, log=False, return_matrix=True)
        # print(cost_now[Pi['G'] > 1e-6].max())
        lambda_vec[j] = np.percentile(cost_now[Pi['G'] > 1e-6], 99)

    lambda_val = lambda_vec.mean()
    return lambda_val

def ROBOT(X=None, Y=None, lambda_val=None):
    ########## detection of outlier
    # print(X.shape, Y.shape)
    cost_matrix = e_dist(X, Y)
    # print(cost_matrix.shape)
    cost_matrix_new = np.copy(cost_matrix)
    cost_matrix[cost_matrix > lambda_val] = lambda_val

    # print(samples_no, cost_matrix.shape)

    Pival, Pi = ot.emd2(np.ones(samples_no)/samples_no, np.ones(samples_no)/samples_no, cost_matrix, processes=4, numItermax=1e6, log=False, return_matrix=True)
    Pimat = Pi['G']
    s1 = np.zeros(samples_no)
    for i in range(samples_no):
        s1[i] = Pimat[i, np.where(cost_matrix_new[i, :] > lambda_val)[0]].sum()

    Xhat_label = np.zeros(samples_no)
    Xhat_label[np.where(s1 > 1e-6)[0]] = 1

    return Xhat_label, s1


def GTOT(X=None, Y=None, lambda_val=None, delta=0.1, S=1):
    # delta : acceptable additive error
    # q_idx : index to get returned values
    dist = c_dist(X, Y)
    nz = len(X)
    supplies = list(map(float, [1.0] * nz))
    demands = list(map(float, [1.0] * nz))
    gtSolver = Mapping(nz, supplies, demands, dist, delta)
    APinfo = np.array(gtSolver.getAPinfo())

    # Clean and process APinfo data
    clean_mask = (APinfo[:,2] >= 1000)
    APinfo_cleaned = APinfo[clean_mask]
    totalFlow = sum(APinfo_cleaned[:, 2])
    cumFlow = np.cumsum((APinfo_cleaned[:, 2]).astype(int))
    flowProgress = (cumFlow)/(1.0 * totalFlow)

    # Summarise APinfo data by phase

    phase_dual = np.stack((flowProgress, 1.0 * APinfo_cleaned[:,0], np.cumsum(1.0 * APinfo_cleaned[:, 4]),\
                                 1.0 * APinfo_cleaned[:, 4], 1.0 * APinfo_cleaned[:, 6]), axis=-1)

    AP_num = np.unique(phase_dual[:, 1], return_counts=True)[1]
    index_first = np.unique(phase_dual[:, 1], return_index=True)[1]
    index_last = index_first + AP_num - 1
    phase_dual_unique = phase_dual[index_last, :]
    phase_dual_unique = np.concatenate((phase_dual_unique, np.expand_dims(AP_num, axis=-1)), axis=-1)

    smooth_by_flowrate = 0.01
    move_step = nz*smooth_by_flowrate
    x = running_mean(flowProgress, int(move_step))
    ypr = running_mean(APinfo_cleaned[:,4], int(move_step))
    
    ignore_transport = np.where(x > 0.4)
    x_ = x[ignore_transport]
    ypr_ = ypr[ignore_transport]

    try:
        kneedle = KneeLocator(ypr_, x_, S, curve="concave", direction="increasing")
        lambda_pred = kneedle.knee_y # kneedle method

    except:
        print("Found no elbow in curve. Setting manually")
        lambda_pred = 0.8

    kneedle_convex = KneeLocator(x_, ypr_, 1, curve="convex", direction="increasing", online = True)
    kneedle_concave = KneeLocator(x_, ypr_, 1, curve="concave", direction="increasing", online = True)
    plt.plot(x_,ypr_)
    i = 0
    for x_convex, x_concave in zip(kneedle_convex.all_knees, kneedle_concave.all_knees):
        plt.axvline(x = x_convex, ls='--', c='red', label='knee_convex' if i == 0 else "")
        plt.axvline(x = x_concave, ls='--', c='blue', label='knee_concave' if i == 0 else "")
        i += 1
    plt.xlabel('flow progress', fontsize=14)
    plt.ylabel('1st derivative', fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig("outlier_detection_ratio030_sm{}.png".format(smooth_by_flowrate))  

    if lambda_val is not None:
        # continue with given cutoff
        k_cut = lambda_val
    else:
        # automate with kneedle
        k_cut = lambda_pred

    def_l = np.array(gtSolver.getDefLog()).T

    def_l = np.round(def_l)
    def_b = np.sum((def_l[:nz, :]==0).astype(int), axis=0)
    def_ind = np.where(def_b == int(nz * k_cut))

    Xhat_label = def_l[0:nz, def_ind[-1][-1]]
    return Xhat_label.astype(int), def_b.astype(int), k_cut, x, ypr

# def plot_histogram(method="ROBOT", samples_no=None, s1=None, gt=None):
#     ########## Plotting the selected outliers
#     plt.title(method)
#     if method == "ROBOT":
#         plt.hist((1/samples_no - s1), bins = 100)
#     elif method == "GTOT":
#         pass
#     plt.show()
#     plt.savefig(method + "_HIST.png")


# def plot_image_grid(method="ROBOT"):
#     n_col = 30
#     n_row = 5
#     if method == "ROBOT":
#         idx = np.random.choice(np.where(s1 > 1e-6)[0], 150, replace=False)
#     elif method == "GTOT":
#         idx = None


#     outliers = X[idx].reshape(-1, 28, 28)
#     fig = plt.figure(figsize=(n_row, n_col))

#     plt.title(method)
#     grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                      nrows_ncols=(n_row, n_col),  # creates 2x2 grid of axes
#                      axes_pad=0.,  # pad between axes in inch.
#                      )
#     i = 0
#     for ax in grid:
#         img = outliers[i]
#         ax.imshow(img, 'gray')
#         ax.set_axis_off()
#         i += 1

#     plt.show()
#     plt.savefig(method + "_OUTLIERS.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nexp', type=int, default=30)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--esp', type=float, default=0.2)
    parser.add_argument('--S', type=int, default=1)
    parser.add_argument('--delta', type=float, default=0.1)
    args = parser.parse_args()
    print(args)

    NUM_EXPERIMENTS = int(args.nexp)
    samples_no = int(args.n)
    eps = args.esp
    S = int(args.S)
    delta = args.delta

    robot_acc = []
    robot_times = []

    gt_acc = []
    gt_times = []
    gt_k = []
    i = 0
    
    while i < NUM_EXPERIMENTS:
        # LOAD Data
        
        a, a_labels, a_idx, b, b_labels, b_idx = data_prep(eps, samples_no)
        X_label = b_idx

        """
        # a, b -> clean, dirty
        print(np.unique(a_labels))
        print(np.unique(b_labels))

        print(np.unique(a_idx))
        print(np.unique(b_idx))

        print(np.sum(X_label))
        """

        # RUN ROBOT
        tic_robot = time.time()
        # find lambda
        ROBOT_lambda_val = ROBOT_lambda_selection(X=b,Y=a,sample_batch=int(samples_no/2))
        # detect outliers
        ROBOT_label, s1 = ROBOT(X=b, Y=a, lambda_val=ROBOT_lambda_val)
        # calculate accuracy
        ROBOT_accuracy = 1.0 - np.linalg.norm(X_label - ROBOT_label, 1)/(1.0 * samples_no)
        toc_robot = time.time()


        robot_acc.append(ROBOT_accuracy)
        robot_times.append(toc_robot - tic_robot)

        print("ROBOT tooks {} seconds with accuracy {}".format(toc_robot - tic_robot, ROBOT_accuracy))

        # RUN GTOT
        tic_gtot = time.time()
        GTOT_label, def_, k_cut, x, y = GTOT(X=b, Y=a, lambda_val=None, delta=0.1, S=S)

        GTOT_accuracy = 1.0 - np.linalg.norm(X_label - GTOT_label, 1)/(1.0 * samples_no)
        toc_gtot = time.time()

        gt_acc.append(GTOT_accuracy)
        gt_times.append(toc_gtot - tic_gtot)
        gt_k.append(k_cut)
        i += 1

        plt.plot(x, y)
        plt.axvline(x = k_cut, ls='--', c='black', label='knee')
        plt.xlabel('flow progress', fontsize=14)
        plt.ylabel('1st derivative', fontsize=14)
        plt.legend(fontsize=14)
        plt.savefig("mnist_N{}_eps{}_no{}.png".format(samples_no, eps, i+1))
        plt.clf()

        print("GTOT tooks {} seconds with accuracy {} (kcut {})".format(toc_gtot - tic_gtot, GTOT_accuracy, k_cut))
        
    print("ROBOT Took {}({}) seconds with accuracy {}({})".format(np.mean(robot_times), np.std(robot_times), np.mean(robot_acc), np.std(robot_acc)))
    print("GTOT Took {}({}) seconds with accuracy {}({}) k_cut {}({})".format(np.mean(gt_times), np.std(gt_times), np.mean(gt_acc), np.std(gt_acc), np.mean(gt_k), np.std(gt_k)))
    results = np.concatenate((gt_times,gt_k,gt_acc,robot_times,robot_acc), axis=-1)
    np.save('results_n_{}_nexp_{}_eps_{}.npy'.format(samples_no, NUM_EXPERIMENTS, eps), results)
