"""
Library for sparse regression algorithms and their respective hyperparameter tuning
"""
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn import linear_model
import matlab.engine
import matlab
import time
# to add SR3 directory to PATH:
import sys
import os


class Model:
    """Data structure to store all necessary data of each model obtained from hyperparameter tuning"""
    def __init__(self, coeffs, coeffs_normed, terms, algo):
        self.algodict = {'FoBa': 3., 'STRidge': 7., 'SR3': 18., 'Lasso': 40.}  # used for visualization purposes only
        self.name = algo  # sparse regression algorithm used to generate model
        self.coefficients = coeffs  # coefficients of original system
        self.normed_coeffs = coeffs_normed  # coefficients of preconditioned system
        self.terms = terms  # number of terms in model; only for convenience, are non-zero numbers on coefficients
        self.mean_rel_err = self.algodict[algo]  # only used if solution known, else no meaning
        self.mean_abs_err = 2.  # only used if solution known, else no meaning
        self.no_incorrect_terms = 1  # only used if solution known, else no meaning


def analyse_algorithms(R, u_t, inverse_transform, true_weight_vect, algo_list, equation, preconditioner):
    """
    This function generates a plot comparing all predicted models from the sparse regression algorithms given in
    'algo_list' for a given precondition setup.

    :param R: System matrix Theta(u) from PDE-FIND
    :param u_t: u_t approximation from PDE-FIND
    :param inverse_transform: inverse transformation matrix
    :param true_weight_vect: Vector containing the true weights; needs same layout as the response vector
    :param algo_list: list containing all strings of sparse regression algorithms to compare
    :param equation: String of data generating equation; only used to generate a unique save-string
    :param preconditioner: String of preconditioner; only used to generate a unique save-string
    :return: None
    """

    # Generate the model list for each sparse regression algorithm
    algo_model_list_list = []
    for algo_index, algo in enumerate(algo_list):
        model_list = generate_models(R, u_t, inverse_transform, algo)
        algo_model_list_list.append(model_list)

    # set up plot
    symbol_list = ['v', 's', 'o', '+', '*', 'D']
    color_list = [(162 / 256., 173 / 256., 0., 0.9), (156 / 256., 157 / 256., 159 / 256., 0.8),
                  (0., 101 / 256., 189 / 256., 1.), (227 / 256., 114 / 256., 34 / 256., 0.9)]
    fig, ax = plt.subplots(figsize=(1.8 * 2.6, 1.8 * 1.95))

    # iterate all models of all algorithms and plot the summaries
    for algo_index, algo in enumerate(algo_list):
        model_list = algo_model_list_list[algo_index]
        for i_model, model in enumerate(model_list):
            set_model_correctness_and_error(model, true_weight_vect)  # calculate absolute and relative error
            # plot model properties
            ax.semilogy(model.terms, model.mean_rel_err, marker=symbol_list[algo_index], markerfacecolor='none',
                        linestyle='None', color=color_list[algo_index], label=algo if i_model == 0 else "")

    # finish plot
    ax.plot((-1., 10.), (1.5, 1.5), color='k')
    # xlim needs to be adjusted with respect to the maximum number of correct term in the model
    ax.set(ylim=[1.e-9, 75.], xlim=[0.5, 6.5], xlabel='Number of Terms', ylabel='MRE')
    ax.legend(loc=4)
    ax.set_yticks(ax.get_yticks()[:-3])
    ax.set_xticks(np.arange(1, 7, 1))
    plt.savefig('Regression_accuracy_comparison_' + str(equation) + '_' + preconditioner + '_test.png')
    plt.savefig('Regression_accuracy_comparison_' + str(equation) + '_' + preconditioner + '_test.pgf')


def generate_models(R, u_t, inverse_transform, algo):
    """
    Function to iterate all hyperparameters of a sparse regression algorithm to generate a set of models

    :param R: System matrix Theta(u) from PDE-FIND
    :param u_t: u_t approximation from PDE-FIND
    :param inverse_transform: inverse transformation matrix to calculate the weights in the original system
    :param algo: sparse regression algorithm for model generation
    :return: model_list containing a predicted models
    """
    model_list = []
    it_max = 10000  # maximum number of iterations after which the Lasso and SR3 are stopped to save computational time
    # in our experience, if the model converges at all, this is usually far sooner than 10000 iterations
    tol_iterativ = 10 * np.finfo(float).eps  # convergence tolerance of SR3 and Lasso
    if algo == 'FoBa':
        log_epsilon_range = np.arange(-15., 15., 0.5)
        for log_epsilon in log_epsilon_range:
            w = FoBa(R, u_t, epsilon=10 ** log_epsilon, backwards_freq=1, maxit_f=20)
            initialize_model(w, model_list, algo, inverse_transform)

    elif algo == 'Lasso':
        log_lambda_range = np.arange(-15., 15., 0.5)  # l1 factor
        for log_lambda in log_lambda_range:
            # initialize Lasso model
            clf = linear_model.Lasso(alpha=10**log_lambda, copy_X=True, fit_intercept=True, max_iter=it_max,
                                     normalize=False, positive=False, precompute=False, random_state=None,
                                     selection='cyclic', tol=tol_iterativ, warm_start=False)
            clf.fit(R, u_t)  # fit model
            w = clf.coef_
            initialize_model(w, model_list, algo, inverse_transform)

    elif algo == 'STRidge':
        log_lambda_range = np.arange(-15, 15., 1.)  # l2 factor (Ridge)
        log_tol_range = np.arange(-16, 10., 1.)
        for log_lambda in log_lambda_range:
            for log_tol in log_tol_range:
                w = STRidge(R, u_t, maxit=1000, lam=10**log_lambda, tol=10**log_tol, normalize=2)
                initialize_model(w, model_list, algo, inverse_transform)

    elif algo == 'SR3':
        # Uses python-matlab interface to directly use the original SR3 implementation.
        # Note that setting up the interface can be a bit tricky; if setting up the interface is too much effort,
        # just leave SR3 out of the 'algo_list' in the SITE file.
        t_sr3_start = time.time()
        eng = matlab.engine.start_matlab()
        eng.setup_matlab(nargout=0)
        log_lambda_range = np.arange(-15, 15., 1.)  # l1 factor
        log_kappa_range = np.arange(-5, 6., 1.)
        for log_kappa in log_kappa_range:
            for log_lambda in log_lambda_range:
                R_matlab = matlab.double(R.tolist())
                u_t_matlab = matlab.double(u_t.tolist())
                # iters can be used to check if model converged or it_max was reached
                x, w, iters = eng.sr3(R_matlab, u_t_matlab, 'mode', '0', 'kap', (10**log_kappa).item(), 'lam',
                                     (10**log_lambda).item(), 'itm', it_max, 'tol', tol_iterativ.item(), 'ptf',
                                     45000, nargout=3)
                w = np.asarray(w)
                initialize_model(w, model_list, algo, inverse_transform)
        eng.quit()
        print('Time for evaluation SR3: ', time.time() - t_sr3_start)

    else: raise ('The algorithm ' + str(algo) + ' is not implemented! (or a typo)')
    return model_list


def initialize_model(w, model_list, algo, inverse_transform):
    """Helper function to store all necessary information in data structure 'Model' and append it to model_list."""
    w_retransformed = np.dot(inverse_transform, w)  # re-transform in original coordinates if a scaled R-matrix is used
    # calculate number of terms
    temp_coeffs = w.copy()
    temp_coeffs[temp_coeffs != 0.] = 1.  # all non zero terms
    terms = int(np.sum(temp_coeffs))

    if len(model_list) != 0:  # a previous model for comparison exists
        if not np.array_equal(w_retransformed, model_list[-1].coefficients):  # different model than at last step
            model_list.append(Model(w_retransformed, w, terms, algo))  # save new model
    else:  # always take first model obtained
        model_list.append(Model(w_retransformed, w, terms, algo))  # save new model


def set_model_correctness_and_error(model, true_weight_vect):
    relative_errors = []
    absolute_errors = []
    for i, coeff in enumerate(model.coefficients):
        if coeff == 0.:
            continue  # term not in model --> nothing to do
        else:
            if true_weight_vect[i] == 0.:
                model.no_incorrect_terms = 0  # term should not be there
                continue  #
            else:  # term is correct --> calculate error
                absolute_errors.append(abs(coeff - true_weight_vect[i]))
                relative_errors.append(abs((coeff - true_weight_vect[i]) / true_weight_vect[i]))
    if len(relative_errors) != 0:  # assuming at least one correct term is found
        if model.no_incorrect_terms:  # only calculate it for true models
            model.mean_rel_err = sum(relative_errors) / float(len(relative_errors))
            model.mean_abs_err = sum(absolute_errors) / float(len(absolute_errors))


def calculate_information_criterion(R, u_t, R_test, u_t_test, inverse_transform, inverse_transform_test, algo, x_nodes):
    """
    This function iterates all hyperparameters of a given sparse regression algorithm and calculates BIC for each
    model with respect to the test set 'R_test'. It outputs a list, which contains for each model the respective
    number of terms, the model list index and the BIC value.

    :param R: System Matrix Theta(u) from PDE-FIND for the training set
    :param u_t: u_t approximation from PDE-FIND for the training set
    :param R_test: System Matrix Theta(u) from PDE-FIND for the test set
    :param u_t_test: u_t approximation from PDE-FIND for the training set
    :param inverse_transform: inverse_transform matrix for the training set
    :param inverse_transform_test: inverse_transform matrix for the test set
    :param algo: The string defining which sparse regression algorithm to use
    :param x_nodes: number of nodes in x; used for estimation of n_eff
    :return: evidence_list: [[number of terms in model, list_index, BIC]]
    """
    model_list = generate_models(R, u_t, inverse_transform, algo)  # iterates hyperparameters to generate models
    evidence_list = calculate_bic(R_test, u_t_test, inverse_transform_test, model_list, x_nodes)
    return evidence_list, model_list


def calculate_bic(R, u_t, inverse_transform_test, model_list, x_nodes):
    """Helper function for calculate_information_criterion to calculate BIC for all candidate models;
    presumes n_eff = x_nodes"""

    evidence_list = []  # model evidence, i.e. BIC for each model
    for i_model, model in enumerate(model_list):
        k = model.terms
        if k == 0:  # does not make sense to compute model without any terms
            continue
        # coefficients are normed and correspond to normed R (if norming in preconditioning applies)
        weights = model.coefficients  # unnormed coefficients from training set
        # norm weights found in training to correspond to norming in test data, which is different in general
        weights = np.dot(np.linalg.inv(inverse_transform_test), weights)

        # n_eff:
        n = x_nodes  # presumption

        y_predict = np.dot(R, weights)  # predicted u_t from training weights
        err_vect = u_t.squeeze() - y_predict.transpose()  # residual vector
        err_vect = err_vect.transpose()
        s_2 = np.dot(err_vect.transpose(), err_vect)  # sum of squared residuals
        # BIC:
        log_evidence_BIC = -(n / 2) * np.log(s_2) - (k / 2) * np.log(n)

        evidence_list.append([i_model, log_evidence_BIC])
    return evidence_list


def STRidge(X0, y, lam, maxit, tol, normalize=2):
    """
    The STRidge algorithm from the Paper 'Data-driven discovery of partial differential equations'
    by Rudy et al., 2017; the code is in its original form:

    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column

    :param X0: System matrix
    :param y: Response vector
    :param lam: Hyperparameter of the L2 penalty
    :param maxit: maximum number of thresholding iterations
    :param tol: thresholding value
    :param normalize: Order of the norm of np.linalg.norm; no norming if normalize = 0
    :return: weight vector w
    """

    n, d = X0.shape
    X = np.zeros((n, d), dtype=np.float64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(0, d):
            Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
            X[:, i] = Mreg[i] * X0[:, i]
    else:
        X = X0

    # Get the standard ridge esitmate
    if lam != 0:
        w = np.linalg.lstsq(X.T.dot(X) + lam * np.eye(d), X.T.dot(y), rcond=None)[0]
    else:
        w = np.linalg.lstsq(X, y)[0]
    num_relevant = d
    biginds = np.where(abs(w) > tol)[0]

    # Threshold and continue
    for j in range(maxit):
        if j == maxit - 1:
            warnings.warn('Maxit in STRidge is not large enough and has stopped Regression at this point')

        # Figure out which items to cut out
        smallinds = np.where(abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]

        # If nothing changes then stop
        if num_relevant == len(new_biginds):
            break
        else:
            num_relevant = len(new_biginds)

        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0:
                # if print_results: print "Tolerance too high - all coefficients set below tolerance"
                return w
            else:
                break
        biginds = new_biginds

        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0:
            w[biginds] = \
            np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam * np.eye(len(biginds)), X[:, biginds].T.dot(y), rcond=None)[0]
        else:
            w[biginds] = np.linalg.lstsq(X[:, biginds], y)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

    if normalize != 0:
        return np.multiply(Mreg, w)
    else:
        return w


def FoBa(X, y, epsilon=0.1, maxit_f=100, maxit_b=5, backwards_freq=5):
    """
    Adaptive Forward-Backward greedy algorithm for sparse regression.
    See Zhang, Tom. 'Adaptive Forward-Backward Greedy Algorithm for Sparse Learning with Linear
    Models', NIPS, 2008

    The Code is based on the implementation from the Paper 'Data-driven discovery of partial differential equations'
    by Rudy et al., 2017; as this Code is partially incorrect, we made adjustments according to Zhang, 2008

    :param X: System matrix
    :param y: Response vector
    :param epsilon: Threshold for improvement of residual; if improvement <epsilon: return
    :param maxit_f: Maximum number of forward iterations: Algorithm terminates afterwards without convergence;
           Warning is printed in this case
    :param maxit_b: Maximum number of backward iterations per backward step
    :param backwards_freq: Number of forward steps after which backward steping is attempted
    :return: weight vector w
    """

    # Initializations
    n, d = X.shape
    F = {}  # dict so save features in each step
    F[0] = set()
    w = {}  # dict to save corresponding weight vector
    w[0] = np.zeros((d, 1), dtype=np.float64)
    k = 0
    delta = {}

    for forward_iter in range(maxit_f):  # forward step
        k = k + 1

        zero_coeffs = np.where(w[k - 1] == 0)[0]  # all coefficients, which are zero, to determine next best candidate
        if len(zero_coeffs) == 0: return w[k - 1]  # all coefficients are included in model --> return

        current_coeffs = np.where(w[k - 1] != 0)[0]
        err_after_addition = []
        X_current = X[:, current_coeffs]

        # check which feature reduces residual most
        for i in zero_coeffs:
            X_test = np.concatenate([X_current, X[:, i].reshape(-1, 1)], 1)  # add candidate feature to current best X
            w_best, err, _, _ = np.linalg.lstsq(X_test, y, rcond=None)
            err_after_addition.append(err)  # save residuals

        best_new_trial = np.argmin(err_after_addition)
        best_new_index = zero_coeffs[best_new_trial]  # feature which reduces residual most

        F[k] = F[k - 1].union({best_new_index})  # update currently best features
        w[k] = np.zeros((d, 1), dtype=np.float64)
        # calculate new weight vector; would be slightly more efficient to save results above
        w[k][list(F[k])] = np.linalg.lstsq(X[:, list(F[k])], y, rcond=None)[0]

        # check for break condition
        delta[k] = np.linalg.norm(X.dot(w[k - 1]) - y) - np.linalg.norm(X.dot(w[k]) - y)
        if delta[k] < epsilon: return w[k - 1]  # improvement was smaller than epsilon --> use old weights

        # backward step, do once every few forward steps
        if forward_iter % backwards_freq == 0 and forward_iter > 0:

            for backward_iter in range(maxit_b):

                non_zeros = np.where(w[k] != 0)  # current features, which are candidates for removal
                err_after_simplification = []

                for j in non_zeros[0]:
                    w_simple = np.copy(w[k])
                    w_simple[j] = 0  # remove candidate
                    # calculate residual for simpler model
                    err_after_simplification.append(np.linalg.norm(X.dot(w_simple) - y))
                least_important = np.argmin(err_after_simplification)
                i_least_important = non_zeros[0][least_important]  # least important feature
                w_simple = np.copy(w[k])
                w_simple[i_least_important] = 0

                # check for break condition on backward step
                delta_p = err_after_simplification[least_important] - np.linalg.norm(X.dot(w[k]) - y)
                if delta_p > 0.5 * delta[k]: break  # do not delete if residual increased too much

                k = k - 1
                F[k] = F[k + 1].difference({i_least_important})  # delete least important feature
                w[k] = np.zeros((d, 1), dtype=np.float64)
                w[k][list(F[k])] = np.linalg.lstsq(X[:, list(F[k])], y, rcond=None)[0]  # recalculate w

    warnings.warn('FoBa stopped, because maxit_f was reached')
    print('epsilon =', epsilon)
    return w[k]


def FoBaGreedy_orig_brunton(X, y, epsilon=0.1, maxit_f=100, maxit_b=5, backwards_freq=5):
    """
    Original Code from the Paper 'Data-driven discovery of partial differential equations' by Rudy et al., 2017
    Incorrect implementation details are highlighted by in-line comments

    Forward-Backward greedy algorithm for sparse regression.

    See Zhang, Tom. 'Adaptive Forward-Backward Greedy Algorithm for Sparse Learning with Linear
    Models', NIPS, 2008
    """

    n, d = X.shape
    F = {}
    F[0] = set()
    w = {}
    w[0] = np.zeros((d, 1), dtype=np.float64)
    k = 0
    delta = {}

    for forward_iter in range(maxit_f):

        k = k + 1

        # forward step
        non_zeros = np.where(w[k - 1] == 0)[0]  # misleading name: should be named zeros
        err_after_addition = []
        residual = y - X.dot(w[k - 1])
        # range is incorrect: needs to iterate all zero-coefficients; otherwise index is wrong
        for i in range(len(non_zeros)):
            alpha = X[:, i].T.dot(residual) / np.linalg.norm(X[:, i]) ** 2
            w_added = np.copy(w[k - 1])
            w_added[i] = alpha
            err_after_addition.append(np.linalg.norm(X.dot(w_added) - y))
        i = np.argmin(err_after_addition)

        F[k] = F[k - 1].union({i})
        w[k] = np.zeros((d, 1), dtype=np.float64)
        w[k][list(F[k])] = np.linalg.lstsq(X[:, list(F[k])], y)[0]

        # check for break condition
        delta[k] = np.linalg.norm(X.dot(w[k - 1]) - y) - np.linalg.norm(X.dot(w[k]) - y)
        if delta[k] < epsilon: return w[k - 1]

        # backward step, do once every few forward steps
        if forward_iter % backwards_freq == 0 and forward_iter > 0:

            for backward_iter in range(maxit_b):

                non_zeros = np.where(w[k] != 0)
                err_after_simplification = []
                for j in range(len(non_zeros)):  # range is wrong: NOT compatible to w; wrong indices
                    w_simple = np.copy(w[k])
                    w_simple[j] = 0
                    err_after_simplification.append(np.linalg.norm(X.dot(w_simple) - y))
                j = np.argmin(err_after_simplification)
                w_simple = np.copy(w[k])
                w_simple[j] = 0

                # check for break condition on backward step
                delta_p = err_after_simplification[j] - np.linalg.norm(X.dot(w[k]) - y)
                if delta_p > 0.5 * delta[k]: break

                k = k - 1;
                F[k] = F[k + 1].difference({j})
                w[k] = np.zeros((d, 1), dtype=np.float64)
                w[k][list(F[k])] = np.linalg.lstsq(X[:, list(F[k])], y)[0]

    return w[k]