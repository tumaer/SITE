"""
This is the main file containing the Sparse Identification of Truncation Errors (SITE) approch
and a few helper functions. In its current form, the data is generated on the fly by the solvers implemented in
'Solvers'. To apply the approach to custom data, the Simulation dat needs to be plugged in at the position, where the
solver is called. This file focuses on analysis of the properties of SITE. Therefore, the
"true" weights corresponding to the analytic weights need to be given in order to be able to generate the plots shown
in the paper. SITE works without the analytic weights as well. However in ths case, the visualization routines are not
as helpful.
"""
import numpy as np
import pickle
import warnings
import time

import Solver
import PDE_FIND_lib as Find
import Sparse_regression_algorithms
import Initialization_Creator as Spline_init
from Initialization_Creator import curve_options, sim_options  # necessary to pickle NURBS curve


np.random.seed(12345)


def site(equation, x_nodes, t_steps, D, P, combinations, optimize_spline=False, x_min=0., x_max=1., acc_space=8,
         acc_time=8, preconditioner='norm', a=1., cfl=0.1, n_ctr_train=15, n_ctr_test=11, curve_degree=8,
         eval_points_per_node=30, bound_amplitude=1., particles=50, iterations=100, c1=0.5, c2=0.3, w_pso=0.9,
         comparison_sparse_regression_algorithms=False, use_spline=True, hotime=0, BIC_algo = 'FoBa'):
    """
    The SITE approach to identify modified differential equations (MDEs) of partial differential equations from
    simulation data. A detailed description of the approach and its properties is given in the paper 'Sparse
    Identification of Trunction Errors' of Thaler, Paehler and Adams, 2019.

    :param equation: String of equation to be analyzed
    :param x_nodes: Number of grid points in space
    :param t_steps: Number of time steps to be used in simulation
    :param D: highest derivative to be included in library
    :param P: highest polynomial order to multiply u with derivative basis functions
    :param combinations: cumulative orders to include in the library
    :param optimize_spline: Will re-optimize the spline if True; else load from disc (default=False)
    :param x_min: left bound of simulation domain Omega (default=0.0)
    :param x_max: right bound of simulation domain Omega (default=1.0)
    :param acc_space: accuracy orders of finite difference stencils in space (default=8)
    :param acc_time: accuracy orders of finite difference stencils in time(default=8)
    :param preconditioner: String defining precondition method (default=norm)
    :param a: Advection speed; Only used if 'equation' == 'Advection' (default=1.0)
    :param cfl: desired cfl number of initial condition (default=0.1)
    :param n_ctr_train: number of NURBS control points within Omega for the training spline (default=15)
    :param n_ctr_test: number of NURBS control points within Omega for the test spline (default=10)
    :param curve_degree: degree of NURBS; should be high enough to ensure high order differentiability (default=8)
    :param eval_points_per_node: number of points of spline per grid node used for interpolation (default=30)
    :param bound_amplitude: maximum allowed y-value of each control point (default=1.0)
    :param particles: number of particles for particle swarm optimization (default=50)
    :param iterations: number of iterations for particle swarm optimization (default=100)
    :param c1: particle swarm optimization parameter c1 (default=0.5)
    :param c2: particle swarm optimization parameter c2 (default=0.3)
    :param w_pso: particle swarm optimization parameter w (default=0.9)
    :param comparison_sparse_regression_algorithms: If True compares the performance of sparse regression algorithms
                                                    (default=False)
    :param use_spline: If True uses spline initialization as IC; else uses a Gaussian IC  (default=True)
    :param hotime: Highest order time derivative to append; No higher order time derivatives if 'hotime'=0 (default=0)
    :param BIC_algo: String defining which sparse regression algorithm to use as basis of model selection using BIC
                     (default='FoBa')
    :return: BIC_model, best_model, rhs_description; model of class 'Model' selected by BIC, model selected by optimal
                                                     choice, List containing the description of the individual terms
    """

    # grid properties:
    dx = (x_max - x_min) / x_nodes
    x_values = np.arange(x_min, x_max, dx)

    # select initial condition (IC) of the simulation
    if use_spline:  # spline initialization
        # needs to be fine, otherwise introduces small discontinuities --> larger FD error
        evaluation_size = eval_points_per_node * x_nodes
        ti = time.time()
        # IC for training set
        init_train = spline_init('train', n_ctr_train, x_values, x_min, x_max, x_nodes, t_steps, equation,
                                 optimize_spline, cfl=cfl, a=a, D=D, P=P, combinations=combinations,
                                 acc_space=acc_space, acc_time=acc_time, curve_degree=curve_degree, hotime=hotime,
                                 evaluation_size=evaluation_size, c1=c1, c2=c2, w_pso=w_pso,
                                 bound_amplitude=bound_amplitude, particles=particles, iterations=iterations)
        # IC for test set
        init_test = spline_init('test', n_ctr_test, x_values, x_min, x_max, x_nodes, t_steps, equation, optimize_spline,
                                cfl=cfl, a=a, D=D, P=P, combinations=combinations, acc_space=acc_space,
                                acc_time=acc_time, curve_degree=curve_degree, hotime=hotime,
                                evaluation_size=evaluation_size, c1=c1, c2=c2, w_pso=w_pso,
                                bound_amplitude=bound_amplitude, particles=particles, iterations=iterations)
        print('Time for spline initialization: ', str((time.time() - ti) / 60.), 'min')

    else:  # Gaussian IC
        import sympy
        x = sympy.symbols('x')
        f_train = sympy.exp(-50 * (x - 0.5) ** 2) + sympy.exp(-50 * (x - 1.5) ** 2) + sympy.exp(-50 * (x + 0.5) ** 2)
        f_test = sympy.exp(-30 * (x - 0.5) ** 2) + sympy.exp(-30 * (x - 1.5) ** 2) + sympy.exp(-30 * (x + 0.5) ** 2)
        lam_f_train = sympy.lambdify(x, f_train, modules=['numpy'])
        lam_f_test = sympy.lambdify(x, f_test, modules=['numpy'])
        init_train = lam_f_train(x_values)
        init_test = lam_f_test(x_values)

    # build training and test set linear system from simulation data using PDE-FIND
    R_train, u_t_train, inverse_transform_train, rhs_description, domain = build_preconditioned_system(
                                    init_train, preconditioner, equation, x_nodes, t_steps, cfl=cfl, x_min=x_min,
                                    x_max=x_max, a=a, acc_space=acc_space, acc_time=acc_time, D=D,
                                    P=P, combinations=combinations, hotime=hotime)

    R_test, u_t_test, inverse_transform_test, _, _ = build_preconditioned_system(
                                    init_test, preconditioner, equation, x_nodes, t_steps, cfl=cfl, x_min=x_min,
                                    x_max=x_max, a=a, acc_space=acc_space, acc_time=acc_time, D=D,
                                    P=P, combinations=combinations, hotime=hotime)

    # calculate the "true" weights corresponding to the analytically derived weights:
    # be careful that the weights are indeed included in the library, otherwise it will through an error
    # used for evaluation of the regression accuracy
    true_weight_vect = get_true_weights(rhs_description, equation, domain, a)

    if comparison_sparse_regression_algorithms:
        algo_list = ['Lasso', 'SR3', 'FoBa', 'STRidge']  # if the MATLAB interface is set up; used in paper
        # algo_list = ['Lasso', 'FoBa', 'STRidge']  # leave out SR3 if the MATLAB interface does not work
        # compare various sparse regression algorithms and plot the results
        Sparse_regression_algorithms.analyse_algorithms(R_train, u_t_train, inverse_transform_train, true_weight_vect,
                                                        algo_list, equation, preconditioner)

    # iterate the sparse regression hyperparamerter of the 'BIC_algo' and calculate BIC with respect to the test set
    # for each of the resulting models:
    evidence_list, model_list = Sparse_regression_algorithms.calculate_information_criterion(
        R_train, u_t_train, R_test, u_t_test, inverse_transform_train, inverse_transform_test, BIC_algo, x_nodes)

    # save list data in numpy array
    evidences = np.zeros((len(evidence_list), 2))
    for i, evidence_instance in enumerate(evidence_list):
        evidences[i, :] = evidence_instance

    # selects model that maximizes BIC, get the model index and retrieve that model from the model list
    BIC_model = model_list[int(evidences[np.argmax(evidences[:, 1]), 0])]

    # print the true MDE and the MDE predicted by FoBa and selected by BIC:
    print('\nactual MDE:')
    Find.print_pde(true_weight_vect, rhs_description)
    print('predicted MDE by FoBa and BIC:')
    Find.print_pde(BIC_model.coefficients, rhs_description)  # prints predicted MDE of current setup

    # Selects the optimal model from a set of models. The optimal model is defined to be the
    # model with the maximum number of correct terms while not including any incorrect ones:
    optimal_model = optimal_choice(model_list, true_weight_vect)

    return BIC_model, optimal_model, rhs_description


def spline_init(name, n_ctr, x_values, x_min, x_max, x_nodes, t_steps, equation, optimize_spline=False, cfl=0.1, a=1.,
                D=6, P=6, combinations=None, hotime=0, acc_space=8, acc_time=8, curve_degree=8, evaluation_size=2000,
                c1=0.5, c2=0.3, w_pso=0.9, bound_amplitude=1., particles=50, iterations=100):
    """
    Wrapper function to create IC from NURBS that is optimized to reduce RMS-VIF of the system.
    Ether loads optimized spline from data or optimizes is again. 'name' specifys if the training or test set
    is chosen (only difference is in save-string used for saving/loading the data).
    """

    if optimize_spline:  # re-optimize spline (can be computationally expensive)
        curve, curve_param, sim_param = Spline_init.create_initialization(x_min, x_max, x_nodes, t_steps, cfl, equation,
                                                                          a, D, P, acc_space, acc_time, n_ctr=n_ctr,
                                                                          curve_degree=curve_degree,
                                                                          evaluation_size=evaluation_size, c1=c1, c2=c2,
                                                                          w=w_pso, bound_amplitude=bound_amplitude,
                                                                          particles=particles, iterations=iterations,
                                                                          combinations=combinations, hotime=hotime)
        # save optimized spline
        pickle.dump([curve, curve_param, sim_param], open('Saved_Data/' + equation + "_" + name + "_spline.bin", "wb"))
    else:  # use saved spline
        curve, curve_param, sim_param = pickle.load(open('Saved_Data/' + equation + "_" + name + "_spline.bin", 'rb'))

    # create IC:
    curve.sample_size = evaluation_size
    init = Spline_init.calculate_y_values(curve, x_values)  # interpolation of spline onto grid

    return init


def build_preconditioned_system(init, preconditioner, equation, x_nodes, t_steps, cfl=0.1, x_min=0., x_max=1., a=1.,
                                acc_space=8, acc_time=8, D=6, P=6, combinations=None, hotime=0):
    """
    Builds a linear system from simulation data using PDE-FIND and preconditions the matrix accordingly.
    Returns the preconditioned system matrix 'R', the corresponding response vector 'u_t', the inverse_transform
    to retrieve the original coefficients from the preconditioned system, the coefficient description and the
    simulation data 'domain' saved in the class 'FDgrid'.
    """

    domain = Solver.solve(x_nodes, t_steps, init, equation, cfl=cfl, a=a, x_min=x_min, x_max=x_max)  # run solver
    Solver.visualize(domain)  # visualize solution
    # apply PDE-FIND algorithm to generate the raw system
    u_t_raw, R_raw, rhs_description = Find.build_linear_system_FD(domain.grid[domain.i_ghost_l + 1:domain.i_ghost_r, :],
                                                                  domain.dt, domain.dx, D=D, P=P, acc_space=acc_space,
                                                                  acc_time=acc_time, order_combinations=combinations,
                                                                  high_order_time_derivs=hotime)
    R, u_t, inverse_transform = Find.precondition_problem(preconditioner, R_raw, u_t_raw)  # precondition the system
    return R, u_t, inverse_transform, rhs_description, domain


def get_true_weights(rhs_description, equation, domain, a=1.):
    """
    Calculates the "true" weights corresponding to the analytically derived weights for the considered problems.
    Be careful that the weights are indeed included in the library, otherwise it will through an error. The
    'true_weight_vect' will correspond to the same coefficients as the vector derived from the sparse regression
    algorithms.
    """

    true_weight_vect = np.zeros(len(rhs_description))

    if equation == 'Advection':
        true_weight_vect[rhs_description.index('u_{x}')] = -a
        true_weight_vect[rhs_description.index('u_{xx}')] = a * domain.dx * 0.5 - (a ** 2) * domain.dt * 0.5
        true_weight_vect[rhs_description.index('u_{xxx}')] = (-domain.dx ** 2) * a / 6 + a ** 2 * domain.dx * \
                                                             domain.dt * 0.5 - a ** 3 * domain.dt ** 2 / 3
        true_weight_vect[rhs_description.index('u_{xxxx}')] = -(-a * domain.dx ** 3 / 24 + a ** 2 * domain.dx ** 2 *
                                                                domain.dt * 7 / 24 - a ** 3 * domain.dx * domain.dt **
                                                                2 / 2 + a ** 4 * domain.dt ** 3 / 4)
        true_weight_vect[rhs_description.index('u_{xxxxx}')] = -(a * domain.dx ** 4 / 120 - a ** 2 * domain.dx ** 3 *
                                                                 domain.dt / 8 + a ** 3 * domain.dx ** 2 * domain.dt **
                                                                 2 * 5 / 12 - a ** 4 * domain.dx * domain.dt ** 3 / 2 +
                                                                 a ** 5 * domain.dt ** 4 / 5)
        true_weight_vect[rhs_description.index('u_{xxxxxx}')] = -(-a * domain.dx ** 5 / 720 + 31 * a ** 2 * domain.dx **
                                                                  4 * domain.dt / 720 - a ** 3 * domain.dx ** 3 *
                                                                  domain.dt ** 2 / 4 + 13 * a ** 4 * domain.dx ** 2 *
                                                                  domain.dt ** 3 / 24 - a ** 5 * domain.dx * domain.dt
                                                                  ** 4 / 2 + a ** 6 * domain.dt ** 5 / 6)

    elif equation == 'Burgers':
        true_weight_vect[rhs_description.index('uu_{x}')] = -1.
        true_weight_vect[rhs_description.index('u_{x}u_{x}u_{x}')] = -domain.dx * domain.dt / 4.
        true_weight_vect[rhs_description.index('uu_{x}u_{x}u_{x}')] = domain.dt ** 2 / 2.
        true_weight_vect[rhs_description.index('u^2u_{x}u_{xx}')] = domain.dt ** 2
        true_weight_vect[rhs_description.index('uu_{x}u_{xx}')] = -domain.dt * domain.dx / 2.
        true_weight_vect[rhs_description.index('u_{x}u_{xx}')] = -domain.dx ** 2 / 2.
        true_weight_vect[rhs_description.index('uu_{xxx}')] = -domain.dx ** 2 / 6.
        true_weight_vect[rhs_description.index('u^3u_{xxx}')] = domain.dt ** 2 / 6.

    elif equation == 'KdV':
        true_weight_vect[rhs_description.index('uu_{x}')] = -6.
        true_weight_vect[rhs_description.index('u_{xxx}')] = -1.
        true_weight_vect[rhs_description.index('u_{ttt}')] = -domain.dt ** 2 / 6.
        true_weight_vect[rhs_description.index('u_{xxxxx}')] = -domain.dx ** 2 / 4.
        true_weight_vect[rhs_description.index('uu_{xxx}')] = -domain.dx ** 2
        true_weight_vect[rhs_description.index('u_{x}u_{xx}')] = -2. * domain.dx ** 2
        true_weight_vect[rhs_description.index('u_{xxxxxxx}')] = -domain.dx ** 4 / 40.
        true_weight_vect[rhs_description.index('uu_{xxxxx}')] = - domain.dx ** 4 / 20.

    else:
        warnings.warn('True vector for specified equation not implemented')

    return true_weight_vect


def optimal_choice(model_list, true_weight_vect):
    """Helper function selects the optimal model from a set of models. The optimal model is defined to be the
    model with maximum number of correct terms while not including and incorrect ones."""
    nr_correct_terms = 0
    best_model = None
    for model in model_list:  # set all models --> also extends to BIC and local EB
        Sparse_regression_algorithms.set_model_correctness_and_error(model, true_weight_vect)
        if model.no_incorrect_terms == 1 and model.terms > nr_correct_terms:  # model is correct and has more terms
            best_model = model
            nr_correct_terms = model.terms
    return best_model
