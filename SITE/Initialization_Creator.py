"""
Implements all functionality of spline initialization using non-uniform rational basis splines (NURBS),
including Particle Swarm Optimization, interpolation functions and calculation of RMS-VIF from a given IC.
Pseudo-code is given in the Paper.
"""
from geomdl import NURBS
from geomdl import utilities
import numpy as np
import pyswarms as ps
import pickle
import scipy
import datetime

import Solver
import PDE_FIND_lib as Find


class sim_options:
    """
    A class to store all discretization parameters for efficient handling;
    For parameter definitions see the main_file
    """

    def __init__(self, x_min, x_max, x_nodes, t_steps, cfl, k, dx, x_values, D, P, combinations, a, equation, acc_space, acc_time, pairwise, hotime):
        self.x_min = x_min
        self.x_max = x_max
        self.x_nodes = x_nodes
        self.t_steps = t_steps
        self.cfl = cfl
        self.k = k  # not used currently; only here for legacy reasons
        self.dx = dx
        self.x_values = x_values
        self.D = D
        self.P = P
        self.combinations = combinations
        self.a = a
        self.equation = equation
        self.acc_space = acc_space
        self.acc_time = acc_time
        self.pairwise = pairwise  # not used currently; only here for legacy reasons
        self.hotime = hotime


class curve_options:
    """Class to store all parameters of the NURBS"""
    def __init__(self, n_ctr, curve_degree, evaluation_size, x_control, delta_x_control):
        self.n_ctr = n_ctr
        self.curve_degree = curve_degree
        self.evaluation_size = evaluation_size
        self.x_control = x_control
        self.delta_x_control = delta_x_control


def create_initialization(x_min, x_max, x_nodes, t_steps, cfl, equation, a, D, P, acc_space, acc_time, n_ctr=10,
                          curve_degree=8, evaluation_size=1000, c1=0.5, c2=0.3, w=0.9, bound_amplitude=1.,
                          particles=50, iterations=100, combinations=None, hotime=0):
    """
    Main function to optimize a NURBS with respect to RMS-VIF to be used as IC in a simulation.
    Needs all simulation parameters, because the optimization
    needs to build the PDE-FIND system for each function value evaluation.

    :param x_min: Left bound of Omega
    :param x_max: Right bound of Omega
    :param x_nodes: Number of points in domain Omega
    :param t_steps: Number of time steps of simulation
    :param cfl: cfl Number of IC
    :param equation: String of equation that the Spline is to be optimized for
    :param a: Advection speed; Only used if 'equation' == 'Advection'
    :param D: Highest derivative to be included in library
    :param P: Highest polynomial order to multiply u with derivative basis functions
    :param acc_space: Accuracy order of finite difference stencils to be used in space
    :param acc_time: Accuracy order of finite difference stencils to be used in time
    :param n_ctr: Number of NURBS control points within Omega
    :param curve_degree: Degree of NURBS
    :param evaluation_size: Number of evaluation points of NURBS used for interpolation
    :param c1: Particle swarm parameter
    :param c2: Particle swarm parameter
    :param w: Particle swarm parameter
    :param bound_amplitude: Maximum allowed y-value of each control point
    :param particles: Number of particles for particle swarm optimization
    :param iterations: Number of iterations for particle swarm optimization
    :param combinations: List of cumulative orders to include in the library
    :param hotime: Highest order time derivative to appended to library
    :return: The optimized NURBS curve and the corresponding sim_options and curve_options
    """

    # initializations:
    dx = (x_max - x_min) / x_nodes
    x_values = np.arange(x_min, x_max, dx)
    pairwise = 0  # not used currently; only here for legacy reasons
    k = 0  # not used currently; only here for legacy reasons

    # set positions of NURBS control points:
    # make sure last control point is at x_max --> otherwise curve not defined between last ctr and x_max
    delta_x_control = (x_max - x_min) / n_ctr
    # f(x_max) = f(x_min) due to periodicity --> ctr-point at x_max is defined by ctr-point at x_min
    x_control = np.arange(x_min, x_max, delta_x_control)

    # save NURBS and Simulation Parameters for consistency:
    sim_param = sim_options(x_min, x_max, x_nodes, t_steps, cfl, k, dx, x_values, D, P, combinations, a,
                            equation, acc_space, acc_time, pairwise, hotime)
    curve_param = curve_options(n_ctr, curve_degree, evaluation_size, x_control, delta_x_control)

    # set attributes for pyswarms
    max_bound = bound_amplitude * np.ones(n_ctr)  # maximum value that control points may exhibit
    min_bound = - max_bound  # minimum value that control points may exhibit
    bounds = (min_bound, max_bound)

    # Set-up pyswarms hyperparameters
    options = {'c1': c1, 'c2': c2, 'w': w}
    kwargs = {"sim_param": sim_param, "curve_param": curve_param}
    # Initialize instance of Particle Swarm Optimization
    optimizer = ps.single.GlobalBestPSO(n_particles=particles, dimensions=n_ctr, options=options, bounds=bounds)
    # Perform optimization; returns best cost and best control point y-values (x-values are set above)
    best_cost, best_pos = optimizer.optimize(whole_swarm, iters=iterations, verbose=3, print_step=5, **kwargs)
    # save cost history for plotting
    pickle.dump(optimizer.cost_history, open('Saved_Data/Cost_history_' + str(equation) +
                                             '{date:%Y_%m_%d%H_%M_%S}.bin'.format(date=datetime.datetime.now()), "wb"))

    # initialize curve to be used in interpolation later
    curve = initialize_NURBS(curve_param, sim_param, best_pos)  # initialize optimized curve

    return curve, curve_param, sim_param


def whole_swarm(x, sim_param, curve_param):
    """
    Wrapper method for function evaluation of the whole swarm using pyswarms.
    It is a input to the pyswarms optimizer 'optimizer.optimize'.
    It calls the function to be optimized 'objective', which Calculates RMS-VIF in our application, for each particle.

    :param x: The swarm for the optimization; numpy.ndarray of shape (n_particles, dimensions)
    :param sim_param: Instance of sim_param (needed to set up simulation)
    :param curve_param: Instance of curve_param (needed to build NURBS curve)
    :return: The computed loss for each particle; numpy.ndarray of shape (n_particles, )
    """
    n_particles = x.shape[0]
    j = [objective(x[i], sim_param, curve_param) for i in range(n_particles)]
    return np.array(j)


def objective(y_objective, sim_param, curve_param):
    """
    Calculates the objective function (RMS-VIF) given the control point y-values of a given particle

    :param y_objective: control point y-values of the particle
    :param sim_param: Instance of sim_param
    :param curve_param: Instance of curve_param
    :return: RMS-VIF
    """
    curve = initialize_NURBS(curve_param, sim_param, y_objective)  # builds NURBS from the control point y-values
    R = calculate_R_from_curve(curve, sim_param)  # builds the PDE-FIND system matrix from the NURBS IC
    rms_vif = calculate_rms_vif(R)  # Calculate RMS-VIF from the matrix R
    return rms_vif


def initialize_NURBS(curve_param, sim_param, y_objective):
    """Builds NURBS from given control points"""

    # initialization:
    curve = NURBS.Curve()
    curve.degree = curve_param.curve_degree
    ctr_points = []  # assign control points according to current y_vector

    # build control points from the y-values and set additional knots to enforce periodicity:
    additional_nodes = curve.degree + 3  # defined by order of spline defining the width of the local support
    # also needs to force symmetry for area left of x = 0., since this is also used in cubic interpolation
    for i in range(additional_nodes):  # set knots left of domain Omega to enforce left hand side symmetry
        # add points at beginning of spline, that correspond to the right part of the spline to enforce periodicity
        # needs to start from leftmost point to create a proper spline
        ctr_points.append([sim_param.x_min - (additional_nodes - i) * curve_param.delta_x_control,
                           y_objective[-(additional_nodes - i)]])

    for i in range(curve_param.n_ctr):  # set control points within Omega
        ctr_points.append([curve_param.x_control[i], y_objective[i]])

    for i in range(additional_nodes):  # set knots right of domain Omega to enforce right hand side symmetry
        # add points at end of spline, that correspond to the begin of the spline to enforce periodicity
        # f(x_max) = f(x_min) must be enforced
        ctr_points.append([sim_param.x_max + i * curve_param.delta_x_control, y_objective[i]])

    # set up the curve
    curve.ctrlpts = ctr_points
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))  # Auto-generate knot vector
    curve.sample_size = curve_param.evaluation_size  # Set number of evaluation points used in interpolation later
    return curve


def calculate_R_from_curve(curve, sim_param):
    """Returns the linear System from PDE-FIND. It takes the curve and the simulation parameters as inputs, employs the
    curve as IC to the data generating simulation and builds the system matrix from the obtained data"""

    init = calculate_y_values(curve, sim_param.x_values)  # interpolate spline onto the simulation grid to be used as IC
    # Run the simulation
    domain = Solver.solve(sim_param.x_nodes, sim_param.t_steps, init, cfl=sim_param.cfl,
                          x_min=sim_param.x_min, x_max=sim_param.x_max, equation=sim_param.equation, a=sim_param.a)
    # buids the linear system using PDE-FIND
    u_t_new, R_new, rhs_description = Find.build_linear_system_FD(domain.grid[domain.i_ghost_l + 1:domain.i_ghost_r, :],
                                                                  domain.dt, domain.dx, D=sim_param.D, P=sim_param.P,
                                                                  order_combinations=sim_param.combinations,
                                                                  acc_space=sim_param.acc_space,
                                                                  high_order_time_derivs=sim_param.hotime,
                                                                  acc_time=sim_param.acc_time)
    return R_new


def calculate_y_values(curve, x_values):
    """
    Interpolates y_values of spline onto the x-values of the grid using cubic interpolation.

    :param curve: NURBS curve as data structure
    :param x_values: Grid values to interpolate spline on
    :return: interpolated y-values corresponing to the grid x-values
    """
    curve.evaluate()  # calculates the NURBS value for each evaluation point (number defined by 'curve.sample_size')
    nurbs_pointlist = curve.evalpts  # list containing all points from evaluation

    # build interpolation table from the NURBS evaluation points
    x_p = np.zeros(len(nurbs_pointlist))
    f_p = np.zeros(len(nurbs_pointlist))
    for i, point in enumerate(nurbs_pointlist):
        x_p[i] = point[0]
        f_p[i] = point[1]

    # cubic interpolation:
    # more efficient than linear interpolation, because the number of evaluation points for linear interpolation is
    # much larger to obtain the same interpolation accuracy. This accuracy is necessary for high order differentiablity
    # at very high accuracy requirements
    f = scipy.interpolate.interp1d(x_p, f_p, kind='cubic', copy=False, assume_sorted=True)
    return f(x_values)


def calculate_rms_vif(X):
    """
    Calculate RMS-VIF for a matrix
    :param X: System matrix
    :return: RMS-VIF
    """

    # R^2: Coefficient of determination for each parameter as being described by the other parameters in X
    r_2 = np.zeros(X.shape[1]-1)
    for i in range(1, X.shape[1]):
        test_column = X[:, i]  # current feature for which R^2 is to be calculated
        X_rest = X[:, np.arange(X.shape[1]) != i]  # all columns except test column
        fit, _, _, _ = np.linalg.lstsq(X_rest, test_column, rcond=None)  # fit linear regression
        y_fit = np.dot(X_rest, fit)  # approximation to the test column by the regression of the other vectors
        _, _, r, _, _ = scipy.stats.linregress(test_column, y_fit)  # calculate r
        r_2[i-1] = r**2  # save the coefficient of determination for the test column

    vif = 1./(1-r_2)  # VIF formula
    rms_vif = np.sqrt(np.mean(np.multiply(vif, vif)))
    return rms_vif




