import time
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist

import ot
from ot.gromov import gromov_wasserstein
import utils
import partial_comb as pcb


def compute_cost_matrices(P, U, prior, nb_dummies=0):
    """Compute the cost matrices (C = C(x_i, y_j), Cs = C^s(x_i, x_k),
    Ct = C^t(y_j, y_l)) and the weigths (uniform weights only).

    Parameters
    ----------
    P: pandas dataframe, shape=(n_p, d_p)
        Positive dataset

    U: pandas dataframe, shape=(n_u, d_u)
        Unlabeled dataset

    prior: percentage of positives on the dataset (s)

    nb_dummies: number of dummy points, default: no dummies
        (to avoid numerical instabilities of POT)

    Returns
    -------
    numpy.array of shape (n_p, n_u) if d_p=d_u, None otherwise
        inter-domain matrix

    numpy.array of shape (n_p, n_p)
        intra-cost matrix for P dataset

    numpy.array of shape (n_u, n_u)
        intra-cost matrix for U dataset

    numpy.array of len (n_d+n_dummies)
        weigths of the P dataset + dummies

    numpy.array of len (n_u)
        weigths of the U dataset
    """
    # Positive dataset with dummy points
    n_unl_pos = int(U.shape[0]*prior)
    P_ = P.copy()
    P_ = np.vstack([P_, np.zeros((nb_dummies, P.shape[1]))])
 
    # weigths
    mu = (np.ones(len(P_))/(len(P_)-nb_dummies))*(n_unl_pos/len(U))
    if nb_dummies > 0:
        mu[-nb_dummies:] = (1 - np.sum(mu[:-nb_dummies]))/nb_dummies
    else:
        mu = mu / np.sum(mu)
    nu = np.ones(len(U))/len(U)

    # intra-domain
    C1 = sp.spatial.distance.cdist(P_, P_)
    C2 = sp.spatial.distance.cdist(U, U)
    if nb_dummies > 0:
        C1[:, -nb_dummies:] = C1[-nb_dummies:, :] = C2.max()*1e2
        C1[-nb_dummies:, -nb_dummies:] = 0

    # inter-domain
    if P_.shape[1] == U.shape[1]:
        C = sp.spatial.distance.cdist(P_, U)
        if nb_dummies > 0:
            C[-nb_dummies:, :] = 1e2 * C[:-nb_dummies, :].max()
    else:
        C = None
    return C, C1, C2, mu, nu


def pu_w_emd(p, q, C, nb_dummies=1):
    """Compute the transport matrix that solves an EMD in the context of
    partial-Wasserstein (with dummy points) + pu learning (with group
    constraints)

    Parameters
    ----------
    p: array of len (n_p+nb_dummies)
        weights of the source (positives)

    q: array of len (n_u)
        weights of the source (unlabeled)

    nb_dummies: number of dummy points, default: 1
        (to avoid numerical instabilities of POT)

    Returns
    -------
    numpy.array of shape (n_p+nb_dummies, n_u)
        transport matrix
    """
    lstlab = np.array([0, 1])
    labels_a = np.append(np.array([0]*(len(p)-nb_dummies)),
                         np.array([1]*(nb_dummies)))

    def f(G):
        res = 0
        for i in range(G.shape[1]):
            for lab in lstlab:
                temp = G[labels_a == lab, i]
                res += (np.linalg.norm(temp, 1))**0.5
        return res

    def df(G):
        W = np.zeros(G.shape)
        for i in range(G.shape[1]):
            for lab in lstlab:
                temp = G[labels_a == lab, i]
                W[labels_a == lab, i] = 0.5*(np.linalg.norm(temp, 1))**(-0.5)
        return W

    Gc = ot.optim.cg(p, q, C, 1e6, f, df, numItermax=20)
    return Gc


def gwgrad(C1, C2, T):
    """Compute the GW gradient. Note: we can not use the trick of Peyré as
    the marginals may not sum to 1.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    numpy.array of shape (n_p+nb_dummies, n_u)
        gradient
    """
    constC1 = np.dot(C1**2/2, np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1)))
    constC2 = np.dot(np.dot(np.ones(C1.shape[0]).reshape(1, -1), T), C2**2/2)
    constC = constC1 + constC2
    A = -np.dot(C1, T).dot(C2.T)
    tens = constC + A
    return tens*2


def gwloss(C1, C2, T):
    """Compute the GW loss.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    loss
    """
    g = gwgrad(C1, C2, T)*0.5
    return np.sum(g * T)


def pu_gw_emd(C1, C2, p, q, nb_dummies=1, group_constraints=False, G0=None,
              log=False, max_iter=20):
    """Compute the transport matrix that solves an EMD in the context of
    partial-Gromov Wasserstein (with dummy points) + pu learning (with group
    constraints or not)

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    p: array of len (n_p+nb_dummies)
        weights of the source (unlabeled)

    q: array of len (n_u)
        weights of the target (unlabeled)

    nb_dummies: number of dummy points, default: 1
        (to avoid numerical instabilities of POT)

    group_constraints: either we want to enforce groups (default: False)

    G0 : array of shape(n_p+nb_dummies, n_u) (default: None)
        Initialisation of the transport matrix

    log: wether we return the loss of each iteration (default: False)

    max_iter: maximum number of iterations (default: 20)

    Returns
    -------
    numpy.array of shape (n_p+nb_dummies, n_u)
        transport matrix

    numpy.array of the history of the loss along the iterations
    """

    # initialisation of the transport matrix
    if G0 is None:
        G0 = np.outer(p, q)
        G0[-nb_dummies:] = 0
    else:
        mask = (G0 < 1e-7)
        G0[mask] = 0
        G0[-nb_dummies:] = 0

    loop = True
    it = 0
    t_loss = []

    # step 8 of alg. 1
    C1_grad = C1.copy()
    C1_grad[-nb_dummies:, -nb_dummies:] = np.quantile(C1[:-nb_dummies,
                                                         :-nb_dummies], 0.75)

    C1_nodummies = C1[:-nb_dummies, :-nb_dummies]
    C2_nodummies = C2.copy()

    loss = gwloss(C1_nodummies, C2_nodummies, G0[:C1_nodummies.shape[0],
                                                 :C2_nodummies.shape[0]])

    while loop:
        # Collect the appropriate submatrix
        idx_pos = np.where(np.sum(G0, axis=0) > 1e-7)[0]
        mask = (G0 > 1e-7)

        C2_current = C2[idx_pos, :]
        C2_current = C2_current[:, idx_pos]

        it += 1
        # Compute the gradient of the active set (step 7 of alg.1)
        G0 = G0[:, idx_pos]
        M = gwgrad(C1_nodummies, C2_current, G0[:-nb_dummies])

        # add the dummy poit + negative points (step 8 of alg.1)
        M_emd = np.ones((len(p), len(q))) * np.quantile(M[M > 1e-7], 0.75)

        # step 9 of alg.1
        idx = 0
        for i in idx_pos:
            M_emd[:-nb_dummies, i] = M[:, idx]
            idx += 1
        M_emd[-nb_dummies:] = M_emd.max() * 1e3
        M_emd = np.asarray(M_emd, dtype=np.float64)

        # step 10 of alg.1
        if group_constraints is False:
            Gc, logemd = ot.lp.emd(p, q, M_emd, log=True)
        else:
            Gc = pu_w_emd(p, q, M_emd, nb_dummies)
            logemd = {}
            logemd['warning'] = None

        if logemd['warning']is not None:
            loop = False
            print("Error in the EMD!!!!!!!")

        G0 = Gc

        prevloss = loss
        loss = gwloss(C1_nodummies, C2_nodummies, G0[:C1_nodummies.shape[0],
                                                     :C2_nodummies.shape[0]])
        t_loss.append(loss)
        if it > max_iter:
            loop = False

        if (it > 2) and (np.abs(prevloss - loss) < 10e-15):
            loop = False

    if log:
        return G0, t_loss
    else:
        return G0


def compute_perf_emd(dataset_p, dataset_u, n_pos, n_unl, prior, nb_reps,
                     nb_dummies=1):
    """Compute the performances of running the partial-W for a PU learning
    task on a given dataset several times

    Parameters
    ----------
    dataset_p: name of the dataset among which the positives are drawn

    dataset_u: name of the dataset among which the unlabeled are drawn

    n_pos: number of points in the positive dataset

    n_unl: number of points in the unlabeled dataset

    prior: percentage of positives on the dataset (s)

    nb_resp: number of runs

    nb_dummies: number of dummy points, default: no dummies
        (to avoid numerical instabilities of POT)

    Returns
    -------
    dict with:
        - the class prior
        - the performances of the p-w (avg among the repetitions)
        - the performances of the p-w with group constraints (avg)
        - the list of all the nb_reps performances of the p-w
        - the list of all the nb_reps performances of the p-w with groups
    """
    perfs = {}
    perfs['class_prior'] = prior
    perfs['emd'] = 0
    perfs['emd_groups'] = 0
    perfs_list = {}
    perfs_list['emd'] = []
    perfs_list['emd_groups'] = []
    perfs_list['time'] = []
    start_time0 = time.time()
    for i in range(nb_reps):
        start_time = time.time()
        P, U, y_u = utils.draw_p_u_dataset_scar(dataset_p, dataset_u, n_pos,
                                                n_unl, prior, i)  # seed=i
        Ctot, _, _, mu, nu = compute_cost_matrices(P, U, prior, nb_dummies)
        nb_unl_pos = int(np.sum(y_u))

        transp_emd = ot.emd(mu, nu, Ctot)
        y_hat = np.ones(len(y_u))
        sum_dummies = np.sum(transp_emd[-nb_dummies:], axis=0)
        y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0
        perfs_list['emd'].append(np.mean(y_u == y_hat))
        perfs['emd'] += (np.mean(y_u == y_hat))

        transp_emd_group = pu_w_emd(mu, nu, Ctot, nb_dummies)
        y_hat = np.ones(len(y_u))
        sum_dummies = np.sum(transp_emd_group[-nb_dummies:], axis=0)
        y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0
        perfs_list['emd_groups'].append(np.mean(y_u == y_hat))
        perfs['emd_groups'] +=  (np.mean(y_u == y_hat))
        perfs_list['time'].append(time.time() - start_time)

    perfs['emd'] = perfs['emd'] / nb_reps
    perfs['emd_groups'] = perfs['emd_groups'] / nb_reps
    perfs['time'] = time.time() - start_time0
    return perfs, perfs_list


def compute_perf_GTOT(dataset_p, dataset_u, n_pos, n_unl, prior, nb_reps, estimate_prior=True):
    """Compute the performances of running the partial-W for a PU learning
    task on a given dataset several times

    Parameters
    ----------
    dataset_p: name of the dataset among which the positives are drawn

    dataset_u: name of the dataset among which the unlabeled are drawn

    n_pos: number of points in the positive dataset

    n_unl: number of points in the unlabeled dataset

    nb_resp: number of runs

    Returns
    -------
    dict with:
        - the class prior
        - the performances of the p-w (avg among the repetitions)
        - the performances of the p-w with group constraints (avg)
        - the list of all the nb_reps performances of the p-w
        - the list of all the nb_reps performances of the p-w with groups
    """
    perfs = {}
    perfs_list = {}
    # if methodname == "all":
    #     perfs_list['GTOT'] = np.zeros((nb_reps, 3))
    #     detected_priors = np.zeros((nb_reps, 3))
    #     perfs['GTOT'] = np.zeros((1,3))
    #     perfs['time'] = []
    # else:
    #     perfs['GTOT'] = 0
    #     perfs_list['GTOT'] = []
    #     detected_priors = np.zeros((nb_reps, 3))
    #     perfs['time'] = [] 
    perfs['GTOT'] = 0
    perfs_list['GTOT'] = []
    detected_priors = []
    perfs['time'] = [] 

    start_time = time.time()
    for i in range(nb_reps):
        start_time = time.time()
        P, U, y_u = utils.draw_p_u_dataset_scar(dataset_p, dataset_u, n_pos,
                                                n_unl, prior, i)  # seed=i
        Ctot, _, _, mu, nu = compute_cost_matrices(P, U, prior, 0)
        nb_unl_pos = int(np.sum(y_u))

        transp_emd = ot.emd(mu, nu, Ctot)
        if estimate_prior:
            y_hat, def_, k_cut = pcb.pu_comb_ot(Ctot, X_label=y_u.values, lambda_val=None, delta=0.1, S=1, prior=prior, dataname = dataset_p, iter=i)
            # y_hat, def_, k_cut = pcb.pu_comb_ot(Ctot, X_label=y_u.values, lambda_val=None, delta=0.1, S=1, prior=prior, dataname = dataset_p, methodname = methodname, iter=i)
        else:
            y_hat, def_, k_cut = pcb.pu_comb_ot(Ctot, X_label=y_u.values, lambda_val=prior, delta=0.1, S=1, prior=prior, dataname = dataset_p)
            # y_hat, def_, k_cut = pcb.pu_comb_ot(Ctot, X_label=y_u.values, lambda_val=prior, delta=0.1, S=1, prior=prior, dataname = dataset_p, methodname = methodname)
        
        perfs['time'].append(time.time() - start_time)

        # if methodname == "all":
        #     perfs_list['GTOT'][i,:] = np.mean(y_hat == np.array(y_u)[:,None],0)
        #     perfs['GTOT'] += perfs_list['GTOT'][i,:]
        #     detected_priors[i,:] = k_cut
        # else:
        #     perfs_list['GTOT'].append(np.mean(y_u == y_hat))
        #     perfs['GTOT'] += (np.mean(y_u == y_hat))
        #     detected_priors.append(k_cut)
        perfs_list['GTOT'].append(np.mean(y_u == y_hat))
        perfs['GTOT'] += (np.mean(y_u == y_hat))
        detected_priors.append(k_cut)

    perfs['GTOT'] = perfs['GTOT'] / nb_reps
    # perfs['time'] = time.time() - start_time
    return perfs, perfs_list, detected_priors


def compute_perf_pgw(dataset_p, dataset_u, n_pos, n_unl, prior, nb_reps,
                     nb_dummies=1):
    """Compute the performances of running the partial-GW for a PU learning
    task on a given dataset several times

    Parameters
    ----------
    dataset_p: name of the dataset among which the positives are drawn

    dataset_u: name of the dataset among which the unlabeled are drawn

    n_pos: number of points in the positive dataset

    n_unl: number of points in the unlabeled dataset

    prior: percentage of positives on the dataset (s)

    nb_resp: number of runs

    nb_dummies: number of dummy points, default: no dummies
        (to avoid numerical instabilities of POT)

    Returns
    -------
    dict with:
        - the class prior
        - the performances of the p-gw (avg among the repetitions)
        - the performances of the p-gw with group constraints (avg)
        - the list of all the nb_reps performances of the p-gw
        - the list of all the nb_reps performances of the p-gw with groups
    """

    perfs = {}
    perfs['class_prior'] = prior
    perfs['pgw'] = 0
    perfs['pgw_groups'] = 0
    perfs_list = {}
    perfs_list['pgw'] = []
    perfs_list['pgw_groups'] = []
    perfs['time'] = []

    for i in range(nb_reps):
        start_time = time.time()
        P, U, y_u = utils.draw_p_u_dataset_scar(dataset_p, dataset_u, n_pos,
                                                n_unl, prior, i)  #     seed=i
        Ctot, C1, C2, mu, nu = compute_cost_matrices(P, U, prior, nb_dummies)
        nb_unl_pos = int(np.sum(y_u))

        Ginit = []
        if Ctot is not None:
            Ginit.append(ot.emd(mu, nu, Ctot))  # We can init. with the EMD
        Ginit.append(None)
        _, Cs, _, ps, pt = compute_cost_matrices(P, U, prior, 0)
        Ginit = Ginit + initialisation_gw(ps, pt, Cs, U, prior, 10,
                                          nb_dummies)

        best_loss = 0
        # We test several init (emd if possible, outer product, barycenter)
        # and keep the one that provides the best loss
        for init in Ginit:
            transp_emd, t_loss = pu_gw_emd(C1, C2, mu, nu, nb_dummies,
                                           group_constraints=False, G0=init,
                                           log=True)
            y_hat = np.ones(len(y_u))
            sum_dummies = np.sum(transp_emd[-nb_dummies:], axis=0)
            y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0
            t_loss[-1] = np.mean(y_u == y_hat)
            if t_loss[-1] > best_loss:
                best_loss = t_loss[-1]
                transp_emd_best = transp_emd.copy()

        y_hat = np.ones(len(y_u))
        sum_dummies = np.sum(transp_emd_best[-nb_dummies:], axis=0)
        y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0
        perfs_list['pgw'].append(np.mean(y_u == y_hat))
        perfs['pgw'] += (np.mean(y_u == y_hat))

        transp_emd_group = pu_gw_emd(C1, C2, mu, nu, nb_dummies,
                                     group_constraints=True, G0=None)

        perfs['time'].append(time.time() - start_time)

        y_hat = np.ones(len(y_u))
        sum_dummies = np.sum(transp_emd_group[-nb_dummies:], axis=0)
        y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0
        perfs_list['pgw_groups'].append(np.mean(y_u == y_hat))
        perfs['pgw_groups'] +=  (np.mean(y_u == y_hat))

    perfs['pgw'] = perfs['pgw'] / nb_reps
    perfs['pgw_groups'] = perfs['pgw_groups'] / nb_reps
    # perfs['time'] = time.time() - start_time
    return perfs, perfs_list


def initialisation_gw(p, q, Cs, U, prior=0, nb_init=5, nb_dummies=1):
    """Initialisation of GW. From a barycenter, it gives a first shot for
    the transport matrix

    Parameters
    ----------
    p: array of len (n_p)
        weights of the source (positives !! with no dummies !!)

    q: array of len (n_u)
        weights of the target (unlabeled)

    Cs: array of shape (n_p,n_p)
        intra-source (P) cost matrix (!! no dummies !!)

    U: array of shape (n_u,d_u)
        U dataset

    prior: percentage of positives on the dataset (s)

    n_init: number of atoms on the barycenter

    nb_dummies: number of dummy points, default: 1
        (to avoid numerical instabilities of POT)

    Returns
    -------
    list of numpy.array of shape (n_p+, n_u)
        list of potentialtransport matrix initialisation

    """
    if nb_init > 2:
        res, _, _ = wass_bary_coarsening(nb_init, np.array(U),
                                         pt=np.ones(U.shape[0])/(U.shape[0]))
    else:
        res, _, _ = wass_bary_coarsening(nb_init, np.array(U),
                                         pt=np.ones(U.shape[0])/(U.shape[0]),
                                         pb=[prior, 1-prior])
    idx = []
    l_gamma = []
    for i in range(nb_init):
        idx = np.where(res[i, :] > 1e-5)[0]
        gamma = np.zeros((len(p) + nb_dummies, len(q)))

        Ct_0 = cdist(U.iloc[idx], U.iloc[idx])
        gamma1 = gromov_wasserstein(Cs, Ct_0, p,
                                    np.ones(Ct_0.shape[0]) / Ct_0.shape[0],
                                    'square_loss')
        gamma1 /= np.sum(gamma1)
        for i in range(len(idx)):
            gamma[:-nb_dummies, idx[i]] = gamma1[:, i]
        l_gamma.append(gamma)
    return l_gamma


def wass_bary_coarsening(n, U, pt=None, pb=None):
    """Computation of a GW barycenter

    Parameters
    ----------
    n: number of atoms of the barycenter

    U: array of shape (n_u,d_u)
        U dataset

    pt: array of len (n_u) (default 1/n_u)
        weights of the target (unlabeled)

    pb: array of len (n) (default 1/n)
        weights of the barycenter

    Returns
    -------
    numpy.array of shape (n, n_u)
        transport matrix between the barycenter and U

    numpy.array of shape (n, n)
        features of the barycenter

    numpy.array of shape (1,1)
        gw cost
    """
    if pb is None:
        pb = np.ones(n) / n
    if pt is None:
        pt = np.ones(U.shape[0]) / U.shape[0]

    U = np.asarray(U)
    pb_diag = np.diag(pb)
    T = np.outer(pb, pt)

    cpt = 0
    err = 1
    tol = 1e-9
    max_iter = 100

    while((err > tol) and cpt < max_iter):
        Tprev = T.copy()
        X = np.dot(U.T, T.T).dot(pb_diag)
        M = sp.spatial.distance.cdist(X.T, U)
        if cpt == 0:
            Mmax = M.max()
        T, logemd = ot.emd(pb, pt/np.sum(pt), M/Mmax, log=True)
        if logemd['warning'] is not None:
            print(logemd['warning'])
        err = np.linalg.norm(T - Tprev)
        cpt += 1
    return T, X.T, np.sum(np.sum(M*T))

def compute_a_case(dataset_a, dataset_b, n_unl, n_pos, prior, nb_reps, estimate_prior):
    perfs_otp, perfs_list_otp, detected_priors = compute_perf_GTOT(dataset_a, dataset_b, n_unl, n_pos, prior, nb_reps, estimate_prior)
    avg_perfs_otp =  perfs_otp['GTOT']
    perfs_otp_std = np.std(perfs_list_otp['GTOT'])
    avg_perfs_otp_time = np.mean(perfs_otp['time'])
    avg_perfs_otp_time_std = np.std(perfs_otp['time'])
    avg_detected_priors = np.mean(detected_priors)
    avg_detected_priors_std = np.std(detected_priors)
    print("acc_{}:{}({})".format(dataset_a, avg_perfs_otp, perfs_otp_std))
    print("time_{}:{}({})".format(dataset_a, avg_perfs_otp_time, avg_perfs_otp_time_std))
    print("pr_{}:{}({})".format(dataset_a, avg_detected_priors, avg_detected_priors_std))

    return avg_perfs_otp, perfs_otp_std, avg_perfs_otp_time, avg_perfs_otp_time_std, avg_detected_priors, avg_detected_priors_std

if __name__ == "__main__":
    n_unl = 800
    n_pos = 800
    nb_reps = 10
    nb_dummies = 10

    prior = 0.518
    P, U, y_u = utils.draw_p_u_dataset_scar('mushrooms', 'mushrooms', n_pos,n_unl, prior, 0)
    
    from pulearn import ElkanotoPuClassifier
    from sklearn.svm import SVC
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
    pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
    pu_estimator.fit(X, y)
    print("abc")

    # avg_perfs_mushrooms_otp, perfs_mushrooms_otp_std, avg_perfs_mushrooms_otp_time, avg_perfs_mushrooms_otp_time_std, avg_mushrooms_detected_priors, avg_mushrooms_detected_priors_std = compute_a_case('mushrooms', 'mushrooms', n_unl, n_pos, prior, nb_reps, True)

    # prior = 0.786
    # avg_perfs_shuttle_otp, perfs_shuttle_otp_std, avg_perfs_shuttle_otp_time, avg_perfs_shuttle_otp_time_std, avg_shuttle_detected_priors, avg_shuttle_detected_priors_std = compute_a_case('shuttle', 'shuttle', n_unl, n_pos, prior, nb_reps, True)

    # prior = 0.898
    # avg_perfs_pageblocks_otp, perfs_pageblocks_otp_std, avg_perfs_pageblocks_otp_time, avg_perfs_pageblocks_otp_time_std, avg_pageblocks_detected_priors, avg_pageblocks_detected_priors_std = compute_a_case('pageblocks', 'pageblocks', n_unl, n_pos, prior, nb_reps, True)

    # prior = 0.167
    # avg_perfs_usps_otp, perfs_usps_otp_std, avg_perfs_usps_otp_time, avg_perfs_usps_otp_time_std, avg_usps_detected_priors, avg_usps_detected_priors_std = compute_a_case('usps', 'usps', n_unl, n_pos, prior, nb_reps, True)

    # prior = 0.658
    # avg_perfs_connect4_otp, perfs_connect4_otp_std, avg_perfs_connect4_otp_time, avg_perfs_connect4_otp_time_std, avg_connect4_detected_priors, avg_connect4_detected_priors_std = compute_a_case('connect-4', 'connect-4', n_unl, n_pos, prior, nb_reps, True)

    # prior = 0.394
    # avg_perfs_spambase_otp, perfs_spambase_otp_std, avg_perfs_spambase_otp_time, avg_perfs_spambase_otp_time_std, avg_spambase_detected_priors, avg_spambase_detected_priors_std = compute_a_case('spambase', 'spambase', n_unl, n_pos, prior, nb_reps, True)

    # n_unl = 800
    # n_pos = 800
    # nb_reps = 10
    # nb_dummies = 10
    # prior = 0.1

    # avg_perfs_mnist_otp, perfs_mnist_otp_std, avg_perfs_mnist_otp_time, avg_perfs_mnist_otp_time_std, avg_mnist_detected_priors, avg_mnist_detected_priors_std = compute_a_case('mnist', 'mnist', n_unl, n_pos, prior, nb_reps, True)
    
    # avg_perfs_mnist_color_change_p_otp, perfs_mnist_color_change_p_otp_std, avg_perfs_mnist_color_change_p_otp_time, avg_perfs_mnist_color_change_p_otp_time_std, avg_mnist_color_change_p_detected_priors, avg_mnist_color_change_p_detected_priors_std = compute_a_case('mnist_color_change_p', 'mnist_color_change_p', n_unl, n_pos, prior, nb_reps, True)