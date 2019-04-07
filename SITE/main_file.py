"""
File to control the parameters of the SITE approach and to specify the postprocessing functionality.
The parameters for each equation are the ones used in the paper. All results of the paper
'Sparse Identification of Trunction Errors' of Thaler, Paehler and Adams, 2019 can be replicated only be
setting the appropriate parameters in this control file. The exceptions are the method of manufactured solutions
in the file 'ManufacturedSolutions', the derivations of the analytic modified differential equations (MDEs) in
the respective files and a few plots are generated in 'Postprocessing_Util'. For an understanding of the parameters
below in this file, we assume knowledge from the preprint of our paper.
"""
import SITE
import Postprocessing_Util


if __name__ == '__main__':

    # ###########################      User input     ##########################################################

    # discretization parameters:
    equation = 'Advection'  # other choices: 'Burgers' ; 'KdV'
    a = None  # initialize, such that input exists for Burgers, KdV, will be overwritten in Advection case
    x_min = 0.  # calculation domain Omega = [x_min, x_max]
    x_max = 1.

    # define discretization parameters and library design for each equation separately
    if equation == 'Advection':
        a = 1.
        x_nodes_list = [300]  # default choice
        # x_nodes_list = [200, 300, 400, 500]  # calculate term orders
        # to calculate resolution properties:
        # x_nodes_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 900, 1000]
        t_steps = 17  # 12 steps are padded
        cfl = 0.01

        # library parameters:
        D = 6  # highest derivative to be included in library
        P = 6  # highest polynomial order to multiply u with derivative basis functions
        # cumulative orders to include in the library; for definition see paper or 'findCombinations' in PDE_FIND_lib
        combinations = [1, 2, 3, 4, 5, 6]  # large library
        # combinations = None  # small library
        hotime = 0  # no higher order time drivatives

    elif equation == 'Burgers':
        x_nodes_list = [10000]  # default choice
        # x_nodes_list = [6000, 8000, 10000, 12000]  # calculate term orders
        # to calculate resolution properties:
        # x_nodes_list = [1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 25000]
        t_steps = 17
        cfl = 0.5

        # library parameters:
        D = 3  # highest derivative to be included in library
        P = 3  # highest polynomial order to multiply u with derivative basis functions
        # cumulative orders to include in the library; for definition see paper or 'findCombinations' in PDE_FIND_lib
        combinations = [1, 2, 3]
        hotime = 0  # no higher order time drivatives

    elif equation == 'KdV':
        x_nodes_list = [100]  # default choice
        # x_nodes_list = [87, 100, 112, 125]  # calculate term orders
        # to calculate resolution properties:
        # x_nodes_list = [50, 60, 75, 87, 100, 110, 125, 135, 150, 175]
        t_steps = 19  # 14 are padded
        cfl = 1.e-6

        # library parameters:
        D = 7  # highest derivative to be included in library
        P = 5  # highest polynomial order to multiply u with derivative basis functions
        # cumulative orders to include in the library; for definition see paper or 'findCombinations' in PDE_FIND_lib
        combinations = [2, 3]
        pairwise = 0
        hotime = 3  # higher order time derivatives appended up to 3rd order
    else: raise Exception('Equation not implemented! (or typo)')

    # accuracy orders of finite difference stencils used in PDE-FIND to build the library Theta(u) and u_t:
    acc_time = 8
    acc_space = 8

    # spline parameters:
    # n_ctr: number of NURBS control points within Omega
    n_ctr_train = 15  # needs to be larger than curve_degree + 3 to be able to enforce periodicity
    n_ctr_test = 11  # needs to be larger than curve_degree + 3 to be able to enforce periodicity
    curve_degree = 8  # degree of NURBS; should be high enough to ensure high order differentiability
    # number of points of spline per grid node used to interpolate spline values on grid points
    eval_points_per_node = 30

    # spline optimization parameters:
    bound_amplitude = 1.  # maximum allowed y-value of each control point
    particles = 50  # number of particles for particle swarm optimization
    iterations = 100  # number of iterations for particle swarm optimization
    # default particle swarm parameters (see documentation of pyswarms for its definitions):
    c1 = 0.5
    c2 = 0.3
    w_pso = 0.9

    # Preconditioner choices:
    # 'norm_precondition': scale the system matrix and apply a puffer transformation afterwards
    # 'norm': only scale the system matrix (robust default)
    # 'precondition': applies puffer transform without scaling first (depreciated)
    # None: use system matrix as obtained from PDE-FIND (depreciated)
    preconditioner = 'norm'

    # Initial condition choices:
    use_spline = True  # if True uses spline initialization; else the Gauss initial condition is used
    optimize_spline = False  # if True re-runs the particle swarm optimization of the spline; else loads saved spline

    # Specify which functionality to be used; setting both true does not make a lot of sense:
    # comparison of sparse regression algorithms for given preconditioner and discretization parameters
    comparison_sparse_regression_algorithms = False
    # study of resolution dependency for given preconditioner and sparse regression algorithm 'BIC_algo'
    plot_resolution_dependency = False
    BIC_algo = 'FoBa'  # sparse regression algorithm for resolution dependency and BIC model selection

    # whether to calculate the term orders:
    # the function assumes all models from the optimal choice to have the same non-zero parameters
    calculate_term_orders = False

    # ###############################    End user input     #######################################################

    # Runs SITE for given Parameters:

    # list initializations for evaluation of resolution properties
    best_model_list = []
    BIC_list = []

    for x_nodes in x_nodes_list:
        BIC_model, best_model, rhs_description = SITE.site(
               equation, x_nodes, t_steps, D, P, combinations, optimize_spline=optimize_spline, x_min=x_min,
               x_max=x_max, acc_space=acc_space, acc_time=acc_space, preconditioner=preconditioner, a=a, cfl=cfl,
               n_ctr_train=n_ctr_train, n_ctr_test=n_ctr_test, curve_degree=curve_degree,
               eval_points_per_node=eval_points_per_node, bound_amplitude=bound_amplitude, particles=particles,
               iterations=iterations, c1=c1, c2=c2, w_pso=w_pso,
               comparison_sparse_regression_algorithms=comparison_sparse_regression_algorithms,
               use_spline=use_spline, hotime=hotime, BIC_algo=BIC_algo)

        # save BIC choice and optimal choice to evaluate resolution properties
        BIC_list.append(BIC_model)
        best_model_list.append(best_model)

    #     Postprocessing

    if calculate_term_orders:
        Postprocessing_Util.calculate_orders(best_model_list, x_nodes_list)

    if plot_resolution_dependency:
        Postprocessing_Util.plot_resolution(best_model_list, BIC_list, x_nodes_list, equation, preconditioner, t_steps)
