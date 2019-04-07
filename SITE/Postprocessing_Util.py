"""
This file contains postprocessing routines called from the main_file to calculate the order of identified terms
and to plot the resolution dependency of a sequence of predicted MDEs. Furthermore, if this file is run, it generates
plots of the Method of manufactured solution convergence as well as the optimization process of the particle swarm
optimization. In both cases, it loads stored data with pickle.
"""
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys


# enable pgf plotting
mpl.use("pgf")
pgf_with_custom_preamble = {
    "font.family": "serif",  # use serif/main font for text elements
    "pgf.rcfonts": False,
    "text.usetex": True,  # use inline math for ticks
}
mpl.rcParams.update(pgf_with_custom_preamble)


def calculate_orders(best_model_list, n_x_list):
    """
    Calculates the order of predicted MDE terms from a list of MDEs for varying n_x.
    The predicted orders are printed.
    This function assumes that all models in best_model_list contain the same non-zero coefficients

    :param best_model_list: list if models of the class 'Model' for varying resolutions
    :param n_x_list: list of n_x corresponding to the models in 'best_model_list';
                     Assumed to be sorted with increasing resolution i.e. from coarse to fine
    :return: None
    """

    coeff_mat = np.zeros((len(n_x_list), len(best_model_list[0].coefficients)))  # for storing of coefficients
    coefficient_list = []

    for i, model in enumerate(best_model_list):
        coeff_mat[i, :] = model.coefficients[:, 0]  # store all coefficients of each model

    for i_column in range(coeff_mat.shape[1]):
        if coeff_mat[0, i_column] == 0.: continue  # coefficient not in model
        else:  coefficient_list.append(coeff_mat[:, i_column])  # store each coefficient for varying n_x

    visualize_order(n_x_list, *coefficient_list)  # calculate and print order


def visualize_order(x_node_list, *lists):
    """
    Prints the order of terms for grid refinement ; dimensions of x_node_list and each list need to fit

    :param x_node_list: List of sequence of n_x; dimension needs to match with each list of 'lists';
                        Assumed to be sorted with increasing resolution i.e. from coarse to fine
    :param lists: Each list contains the same term for varying grid resolutions
                  An arbitrary number of lists; Can be lists as well as numpy arrays
    :return: None
    """

    for i_list, current_inputlist in enumerate(lists):
        empirical_orders = np.zeros(len(x_node_list) - 1)  # to save empirical orders
        for i in range(len(x_node_list) - 1):
            # calculate empirical order
            empirical_orders[i] = np.log(current_inputlist[i] / current_inputlist[i + 1]) /\
                                  np.log(x_node_list[i+1]/x_node_list[i])
        print('empirical order for input', i_list, ':', empirical_orders, '; mean:', np.mean(empirical_orders))


def plot_resolution(best_model_list, bic_list, x_nodes_list, equation, preconditioner, t_steps):
    """
    Generates a plot visualizing MAE and MRE as well as the number of terms identified by BIC and by optimal choice
    for varying grid resolution

    :param best_model_list: List of models of class 'Model' from the optimal choice procedure for varying n_x
    :param bic_list: List of models of class 'Model' from the optimal choice procedure for varying n_x
    :param x_nodes_list: List of n_x corresponding to each model from 'best_model_list' and 'BIC_list'
    :param equation: Equation string used to generate a unique save-string
    :param preconditioner: Preconditioner string used to generate a unique save-string
    :param t_steps: Number of time steps of simulation; used to generate a unique save-string
    :return: None
    """

    if len(best_model_list) <= 1: sys.exit('Not useful to plot resolution defendency if only one model is considered')

    quality_mat = np.zeros((len(best_model_list), 4))  # stores information of each model in array
    # quality_mat structure: [Number of terms of optimal choice, Number of terms of BIC choice, MRE, MAE]

    # fill quality_mat
    for i, model in enumerate(best_model_list):
        # if optimal model does not include any correct term; only if sparse regression algorithm proposes
        # not a single correct model
        if model is None:
            quality_mat[i, 2] = np.nan
            quality_mat[i, 3] = np.nan
            quality_mat[i, 0] = 0.
        else:  # optimal model with at least one term exists
            quality_mat[i, 2] = model.mean_rel_err
            quality_mat[i, 3] = model.mean_abs_err
            quality_mat[i, 0] = model.terms
        quality_mat[i, 1] = bic_list[i].terms

    fig_resolution, ax_resolution = plt.subplots(figsize=(6.2, 4.0))

    # Plot connecting line number of terms selected by BIC and optimal choice
    ax_resolution.plot(x_nodes_list, quality_mat[:, 0], color=(227 / 256., 114 / 256., 34 / 256., 0.5), label='Optimal')
    ax_resolution.plot(x_nodes_list, quality_mat[:, 1], color=(138 / 256., 138 / 256., 138 / 256., 0.5), label='BIC')

    for i, model in enumerate(best_model_list):

        # encode model correctness for optimal model
        if model is None:  # not a single term included in model
            num_terms = 0
            correct = 0  # as a model containing 0 terms is incorrect, it is encoded that way
        else:  # if at least one term is included, the model is correct by construction (see site, optimal choice)
            num_terms = model.terms
            correct = 1

        # plot the number of identified terms of each model and encode in the color if it is correct
        ax_resolution.scatter(x_nodes_list[i], num_terms, facecolors='none',
                              color='b' if correct == 1 else 'r', marker='o')
        ax_resolution.scatter(x_nodes_list[i], bic_list[i].terms,
                              color='b' if bic_list[i].no_incorrect_terms == 1 else 'r', marker='x')

    # plot MRE and MAE
    ax_resolution_2 = ax_resolution.twinx()
    ax_resolution_2.semilogy(x_nodes_list, quality_mat[:, 2], color=(162 / 256., 173 / 256., 0., 1.0),
                             label='MRE optimal')
    ax_resolution_2.semilogy(x_nodes_list, quality_mat[:, 3], color=(0., 101 / 256., 189 / 256., 1.),
                             label='MAE optimal')

    # settings
    ax_resolution.set(ylim=[-0.5, 10.5], xlabel='$n^x$', ylabel='Number of Terms')
    ax_resolution_2.set(ylim=[1.e-14, 1.], ylabel='$L_1$ Error')
    ax_resolution_2.legend(loc=4)
    ax_resolution.legend(loc=3)
    plt.savefig('Resolution_properties_' + str(equation) + '_' + str(preconditioner) + str(t_steps) + '.pgf')
    plt.savefig('Resolution_properties_' + str(equation) + '_' + str(preconditioner) + str(t_steps) + '.png')


if __name__ == '__main__':

    plot_optimization_history = True  # plots optimization process of particle swarm optimization
    plot_MMS_convergence = True  # plots convergence of Method of Manufactured solution from saved convergence data

    if plot_optimization_history:
        # cost history is list of cost values; sorted by iteration
        cost_history_train_adv = pickle.load(open('Saved_Data/Cost_history_Advection_Train.bin', 'rb'))
        cost_history_test_adv = pickle.load(open('Saved_Data/Cost_history_Advection_Test.bin', 'rb'))
        cost_history_train_bur = pickle.load(open('Saved_Data/Cost_history_Burgers_Train.bin', 'rb'))
        cost_history_test_bur = pickle.load(open('Saved_Data/Cost_history_Burgers_Test.bin', 'rb'))
        cost_history_train_kdv = pickle.load(open('Saved_Data/Cost_history_KdV_Train.bin', 'rb'))
        cost_history_test_kdv = pickle.load(open('Saved_Data/Cost_history_KdV_Test.bin', 'rb'))

        fig_history, ax_history = plt.subplots(figsize=(4.7, 3.5))
        ax_history.semilogy(cost_history_train_adv, label='Advection', color=(0., 101 / 256., 189 / 256., 1.))
        ax_history.semilogy(cost_history_test_adv, linestyle='--', color=(0., 101 / 256., 189 / 256., 1.))
        ax_history.semilogy(cost_history_train_bur, label='Burgers', color=(227/256., 114/256., 34/256., 1.))
        ax_history.semilogy(cost_history_test_bur, linestyle='--', color=(227/256., 114/256., 34/256., 1.))
        ax_history.semilogy(cost_history_train_kdv, label='KdV', color=(162/256., 173/256., 0., 1.0))
        ax_history.semilogy(cost_history_test_kdv, linestyle='--', color=(162/256., 173/256., 0., 1.0))
        ax_history.legend(loc=3)
        ax_history.set(xlabel='Iteration', ylabel='VIF', ylim=[1., 1.e7])

        plt.savefig('Convergence_Splines.pgf')
        plt.savefig('Convergence_Splines.png')

    if plot_MMS_convergence:

        x_nodes_list = [100, 200, 400, 800]  # evaluation resolutions
        x_annotate = [125, 250, 500]  # positions for plotting of empirical order
        equation_list = ['Advection', 'Burgers', 'KdV']
        color_list = [(0., 101 / 256., 189 / 256., 1.), (227/256., 114/256., 34/256., 1.),
                      (162/256., 173/256., 0., 1.0)]

        fig_MMS_order, ax_MMS_order = plt.subplots(figsize=(4.7, 3.5))

        for eq_index, equation in enumerate(equation_list):
            L0, L2 = pickle.load(open('Saved_Data/MMS_convergence_' + equation + '.bin', 'rb'))  # load MMS errors

            order = np.zeros([len(x_nodes_list) - 1, 2])
            for i in range(len(x_nodes_list) - 1):
                # calculate empirical order
                order[i, 0] = np.log(L0[i] / L0[i + 1]) / np.log(x_nodes_list[i + 1] / x_nodes_list[i])
                order[i, 1] = np.log(L2[i] / L2[i + 1]) / np.log(x_nodes_list[i + 1] / x_nodes_list[i])

            ax_MMS_order.loglog(x_nodes_list, L0, label=equation,
                                color=color_list[eq_index], marker='o')
            ax_MMS_order.loglog(x_nodes_list, L2,
                                linestyle='--', color=color_list[eq_index], marker='o')
            for i_anno, x_anno in enumerate(x_annotate):
                # plot annotation above L0
                ax_MMS_order.annotate(str(round(order[i_anno, 0], 3)),
                                      xy=(x_anno, 1.5 * 0.5*(L0[i_anno + 1] + L0[i_anno])))
                # plot annotation below L2
                ax_MMS_order.annotate(str(round(order[i_anno, 0], 3)),
                                      xy=(x_anno, 0.25 * 0.5*(L2[i_anno + 1] + L2[i_anno])))

        ax_MMS_order.set(xlabel='$n^x$', ylabel='Error', ylim=[5.e-12, 0.2], xlim=[80, 1000])
        plt.legend(loc=0)
        plt.savefig('Convergence_MMS.png')
        plt.savefig('Convergence_MMS.pgf')
