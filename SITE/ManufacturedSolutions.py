"""
Implements the Method of Manufactured Solutions (MMS) for verification of the custom solvers.
The spatial and temporal scales are refined together with a constant cfl number.
"""
import sympy as sp
import numpy as np
import Solver
import pickle
from Postprocessing_Util import visualize_order


def calculate_source_term(f, equation, a=1.):
    """
    Calculates the source term for MMS for the Advection, Burgers' and KdV Equation

    :param f: Sympy function defining the Manufactured Solution
    :param equation: String defining which equation to use
    :param a: Advection velocity; only used if 'equation' == 'Advection'
    :return: Sympy function defining the Source Term
    """

    if equation == 'Advection':
        return f.diff(t) + a * f.diff(x)  # Advection Equation: u_t + a * u_x = 0
    elif equation == 'Burgers':
        return f.diff(t) + f * f.diff(x)  # Burgers Equation: u_t + u * u_x = 0
    elif equation == 'KdV':
        return f.diff(t) + 6. * f * f.diff(x) + f.diff(x, x, x)  # KdV equation: u_t + 6* u * u_x + u_xxx = 0
    else:
        raise Exception('Equation not implemented! (or typo)')


if __name__ == '__main__':
    # manufactured solution parameters: u = u_0 * (sin((2*pi/delta_x) * x + omega * t) + epsilon_bar)
    omega = 2. * sp.pi
    u_0 = 1.
    epsilon_bar = 0.001

    # simulation parameters:
    equation = 'Advection'  # change here which equation is to be verified

    a = 1.  # coefficient of advection equation; only used in this case
    x_min = 0.  # calculation domain Omega = [x_min, x_max]
    x_max = 1.
    delta_x = x_max - x_min
    t_test = 0.1
    if equation == 'KdV':
        t_test = 1.e-8  # otherwise simulation takes too long
    cfl = 0.1
    if equation == 'KdV':
        cfl = 1.e-10  # very small time steps necessary for stability

    # Define Manufactured Solution:
    t = sp.symbols('t')  # define sympy variables
    x = sp.symbols('x')
    # function needs to be periodic for all times, because periodic BC are assumed
    u = u_0 * (sp.sin((2*sp.pi/delta_x)*x + omega * t) + epsilon_bar)  # Manufactured solution
    s = calculate_source_term(u, equation, a=a)  # Source Term to enforce Manufactured Solution

    L0_list = []
    L2_list = []

    x_node_list = [100, 200, 400, 800]  # grid resolutions for refinement

    for x_nodes in x_node_list:

        dx = delta_x / x_nodes  # calculate new cell width for new resolution
        x_values = np.arange(x_min, x_max, dx)  # x = x_max is not included --> equals x(0) due to periodic BC

        # IC:
        init = np.zeros(x_nodes)
        lam_u_init = sp.lambdify(x, u.subs({t: 0.}), modules=['numpy'])  # IC is manufactured solution u(t = 0)
        init[:] = lam_u_init(x_values)

        # calculate maximum convection speed to get time step width dt from cfl
        if equation == 'Advection':
            u_max_convection = abs(a)
        else:  # nonlinear equation
            # |u_max| in first time step; if |u_max| increases during simulation, the cfl number will be slightly larger
            # than defined above; we need constant delta_t, thus we cannot adjust to this
            u_max_convection = np.max(np.abs(init))
        dt = (cfl * dx) / u_max_convection

        n_t = int((t_test/dt) + 1)  # Total number of time steps; first time step is IC --> + 1
        # only integer time steps possible --> for given dt in general not possible to hit t_test exactly
        t_test_exact = (n_t - 1) * dt  # adjust evaluation time of manufactured solution to match last time step exactly

        domain = Solver.solve(x_nodes, n_t, init, equation, cfl=cfl, x_min=x_min, x_max=x_max,
                              a=a, manufactured=True, s=s)  # run solver
        Solver.visualize(domain)  # plot the solution

        # calculate errors
        lam_u_test = sp.lambdify(x, u.subs({t: t_test_exact}), modules=['numpy'])  # exact solution at t_test_exact
        u_true = lam_u_test(x_values)
        err = domain.grid[domain.i_ghost_l + 1:domain.i_ghost_r, -1] - u_true  # dont use BC
        L0_list.append(np.linalg.norm(err, ord=np.inf))
        l2 = np.sqrt(np.mean(np.multiply(err, err)))  # root mean square error
        L2_list.append(l2)

    # prints empirical orders derived from MMS; check if they match the theoretical one
    visualize_order(x_node_list, L0_list, L2_list)

    # save data for plotting in Postprocessing_Util
    pickle.dump([L0_list, L2_list], open('Saved_Data/TEMP_MMS_convergence_' + str(equation) + '.bin', "wb"))
