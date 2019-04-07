"""
Implementation of custom solvers: advection equation with forward-time, backward-space; Burgers' equation with
MacCormack scheme and Korteweg-de Vries equation with Zabusky and Kruska scheme.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sympy as sp
import warnings


# enable pgf printing of solution plot
mpl.use("pgf")
pgf_with_custom_preamble = {
    "font.family": "serif",  # use serif/main font for text elements
    "pgf.rcfonts": False,
    "text.usetex": True,  # use inline math for ticks
}
mpl.rcParams.update(pgf_with_custom_preamble)


class FDgrid:
    """
    Class for initialization of the calculation domain and data storage;
    handles an arbitrary number of ghost cells 'n_ghost'
    """
    def __init__(self, x_nodes, n_t, x_min, x_max, u_max_convection, cfl, n_ghost):
        """
        Initializes the calculation domain

        :param x_nodes: Number of points in domain Omega
        :param n_t: Total number of time steps including IC t = 0
        :param x_min: left bound of Omega
        :param x_max: right bound of Omega
        :param u_max_convection: convection speed to calculate dt from cfl
        :param cfl: cfl number of IC
        :param n_ghost: number of ghost cells for periodic BC needed by scheme (1 for advection and Burgers; 2 for KdV)
        """

        self.n_x = x_nodes + n_ghost * 2  # ghost nodes at both sides
        self.x_nodes = x_nodes
        self.n_t = n_t
        self.x_min = x_min
        self.x_max = x_max
        self.n_ghost = n_ghost
        self.i_ghost_r = x_nodes + n_ghost  # index of leftmost ghost node at right boundary
        self.i_ghost_l = n_ghost - 1  # index of rightmost ghost node at left boundary
        self.dx = (x_max - x_min) / x_nodes  # save spatial width
        self.dt = (cfl*self.dx)/u_max_convection  # set dt according to desired cfl number
        self.t_max = self.dt * (n_t - 1)  # t = 0 is initial condition
        self.grid = np.zeros((self.n_x, n_t), dtype=np.float64)  # initialize array to store simulation results

    def fill_BC(self, i_time):
        """fills ghost cells with periodic boundary conditions"""
        vect_to_set = np.zeros(self.n_x)
        # copies the data within the domain to a vector
        vect_to_set[self.i_ghost_l + 1: self.i_ghost_r] = self.grid[self.i_ghost_l + 1: self.i_ghost_r, i_time]
        vect_to_set = set_periodic_BC(self, vect_to_set)  # sets periodic BCs for vector
        self.grid[:, i_time] = vect_to_set  # copies filled vector back onto the grid


def set_periodic_BC(domain, vect):
    """Helper function called from 'fill_BC' to set the periodic BCs for arbitrary number of ghost cells"""
    for i in range(domain.n_ghost):  # set all values for ghost cells, starting from left for both sides
        # value of left ghost cell is value of most right real cell
        # leftmost left node is to be n_ghost nodes left of the leftmost right ghost node
        vect[i] = vect[domain.i_ghost_r - domain.n_ghost + i]  # set left boundary element
        # leftmost right ghost node is first real left node
        vect[domain.i_ghost_r + i] = vect[i + domain.n_ghost]  # right boundary
    return vect


def solve(x_nodes, n_t, initial_cond, equation, x_min=0., x_max=1., cfl=0.1, a=1., manufactured=False, s=None):
    """

    :param x_nodes: Number of points in domain Omega
    :param n_t: Total number of time steps including IC t = 0
    :param initial_cond: Numpy array containing the values of the IC; Dimension is x_nodes
    :param equation: String of equation to be solved
    :param x_min: left bound of Omega (default=0.0)
    :param x_max: right bound of Omega (default=1.0)
    :param cfl: desired cfl number of IC (default=0.1)
    :param a: Advection speed; Only used if 'equation' == 'Advection' (default=1.0)
    :param manufactured: Whether the Method of Manufactured solution is to be calculated (forcing 's' will be applied)
                         (default=False)
    :param s: Forcing Function of MMS (default=None)
    :return: FDgrid object containing the simulation results in FDgrid.grid and information about the discretization
    """

    # set up calculation domain:
    if equation == 'Advection':
        u_max_convection = a
        if a < 0.: warnings.warn('FTBS only implemented for a > 0: solver will not be stable')
    else:  # for nonlinear equations: calculate maximum convection speed in cfl from initial conditions
        u_max_convection = np.max(np.abs(initial_cond))
    n_ghost = 1  # for FTBS and MacCormack
    if equation == 'KdV':
        n_ghost = 2
    domain = FDgrid(x_nodes, n_t, x_min, x_max, u_max_convection, cfl, n_ghost)  # initializes calculation domain
    domain.grid[domain.i_ghost_l + 1:domain.i_ghost_r, 0] = initial_cond  # set IC
    domain.fill_BC(0)  # sets ghost cells for IC

    # initialize sympy variables to process forcing term used in MMS
    x_values = np.arange(x_min, x_max, domain.dx)  # for evaluation of forcing function
    x = sp.symbols('x')
    t = sp.symbols('t')

    if equation == 'Advection':  # solve advection u_t + a u_x = 0 using FTBS
        for i_t in range(1, n_t):
            for i_x in range(domain.i_ghost_l + 1, domain.i_ghost_r):  # iterate domain without ghost cells
                # FTBS for a > 0:
                # u_i^n+1 = u_i^n - cfl (u_i^n - u_i-1^n) ; cfl = a * t / dx
                domain.grid[i_x, i_t] = domain.grid[i_x, i_t - 1] - cfl * \
                                       (domain.grid[i_x, i_t - 1] - domain.grid[i_x - 1, i_t - 1])
            if manufactured:  # add forcing from MMS
                time = (i_t - 1) * domain.dt  # to evaluate source term for current time step
                domain.grid[domain.i_ghost_l + 1:domain.i_ghost_r, i_t] += domain.dt * calculate_forcing_manufactured(
                                                                                       x_values, x, t, s, time)
            domain.fill_BC(i_t)

    elif equation == 'Burgers':
        # solve Burgers equation u_t + g_x = 0 ; g = u^2/2 using 2nd order scheme in time and space of Mac Cormack
        u_predictor = np.zeros(domain.n_x)  # initialize saving of predictor step
        for i_t in range(1, n_t):
            time = (i_t - 1) * domain.dt  # time for evaluation source term

            # prediction step:
            for i_x in range(domain.i_ghost_l + 1, domain.i_ghost_r):  # iterate domain without ghost cells
                # u_i^n+1_pred = u_i^n - dt/dx(g_i+1^n - g_i^n)
                u_predictor[i_x] = domain.grid[i_x, i_t - 1] - (domain.dt / domain.dx) *\
                                   (0.5 * domain.grid[i_x + 1, i_t - 1] ** 2 - 0.5 * domain.grid[i_x, i_t - 1] ** 2)
            if manufactured:  # add forcing from MMS
                u_predictor[domain.i_ghost_l + 1:domain.i_ghost_r] += domain.dt * calculate_forcing_manufactured(
                                                                                  x_values, x, t, s, time)
            # set periodic BC for predictor; MacCormack only needs a single ghost cell
            u_predictor[domain.i_ghost_l] = u_predictor[domain.i_ghost_r - 1]
            u_predictor[domain.i_ghost_r] = u_predictor[domain.i_ghost_l + 1]

            # correction step:
            for i_x in range(domain.i_ghost_l + 1, domain.i_ghost_r):  # iterate domain without ghost cells
                # u_i^n+1 = u_i^n - 0.5*(dt/dx) * ((g_i+1^n - g_i^n) + (g_i^n_pred - g_i-1^n_pred))
                domain.grid[i_x, i_t] = domain.grid[i_x, i_t - 1] - 0.5 * (domain.dt/domain.dx) * \
                                ((0.5 * domain.grid[i_x + 1, i_t - 1] ** 2 - 0.5 * domain.grid[i_x, i_t - 1] ** 2) +
                                 (0.5 * u_predictor[i_x] ** 2 - 0.5 * u_predictor[i_x - 1] ** 2))
            if manufactured:  # forcing needs to be evaluated at intermediate step
                domain.grid[domain.i_ghost_l + 1:domain.i_ghost_r, i_t] += domain.dt * calculate_forcing_manufactured(
                                                                                x_values, x, t, s, time + 0.5*domain.dt)

            domain.fill_BC(i_t)

    elif equation == 'KdV':
        # solve KdV u_x + 6*uu_x + u_xxx = 0 using the explicit 2nd order scheme in space and time of Zabusky and Kruska

        # use forward time scheme in first time step to generate data to use for central time stepping
        for i_x in range(domain.i_ghost_l + 1, domain.i_ghost_r):
            # u_j^k+1 = u_j^k - (dt/dx)*(u_j+1^k + u_j^k + u_j-1^k) * (u_j+1^k - u_j-1^k) -
            # 0.5 * dt/dx**3 * (u_j+2^k - 2 * u_j+1^k + 2 * u_j-1^k - u_j-2^k)
            domain.grid[i_x, 1] = domain.grid[i_x, 0] - (domain.dt/domain.dx) * (domain.grid[i_x + 1, 0] +
                                  domain.grid[i_x, 0] + domain.grid[i_x - 1, 0]) * 0.5 * (domain.grid[i_x + 1, 0]
                                  - domain.grid[i_x - 1, 0]) - 0.5 * (domain.dt / domain.dx ** 3) * \
                                  (domain.grid[i_x + 2, 0] - 2. * domain.grid[i_x + 1, 0] + 2. * domain.grid[i_x - 1, 0]
                                  - domain.grid[i_x - 2, 0])
        if manufactured:  # add forcing for MMS
            domain.grid[domain.i_ghost_l + 1:domain.i_ghost_r, 1] += domain.dt * calculate_forcing_manufactured(
                                                                                 x_values, x, t, s, 0.)
        domain.fill_BC(1)

        # central time stepping from now on
        for i_t in range(2, n_t):

            for i_x in range(domain.i_ghost_l + 1, domain.i_ghost_r):
                # u_j^k+1 = u_j^k-1 - 2 * (dt/dx) * (u_j+1^k + u_j^k + u_j-1^k) * (u_j+1^k - u_j-1^k) - dt / dx**3 *
                # (u_j+2^k - 2 * u_j+1^k + 2 * u_j-1^k - u_j-2^k)
                domain.grid[i_x, i_t] = domain.grid[i_x, i_t - 2] - 2. * (domain.dt / domain.dx) * \
                                        (domain.grid[i_x + 1, i_t - 1] + domain.grid[i_x, i_t - 1] +
                                        domain.grid[i_x - 1, i_t - 1]) * (domain.grid[i_x + 1, i_t - 1] -
                                        domain.grid[i_x - 1, i_t - 1]) - (domain.dt / (domain.dx ** 3)) * \
                                        (domain.grid[i_x + 2, i_t - 1] - 2. * domain.grid[i_x + 1, i_t - 1] +
                                        2. * domain.grid[i_x - 1, i_t - 1] - domain.grid[i_x - 2, i_t - 1])
            if manufactured:  # add forcing for MMS
                time = (i_t - 1) * domain.dt
                domain.grid[domain.i_ghost_l + 1:domain.i_ghost_r, i_t] += 2. * domain.dt * \
                                                                calculate_forcing_manufactured(x_values, x, t, s, time)
            domain.fill_BC(i_t)

    else: raise Exception('Equation not implemented! (or typo)')

    return domain


def calculate_forcing_manufactured(x_values, x, t, s, time):
    """Calculates the forcing term for MMS from the source term; directly depends on time"""
    lam_s = sp.lambdify(x, s.subs({t: time}), modules=['numpy'])
    return lam_s(x_values)


def visualize(domain):
    """Function to plot the first and last time step of a simulation; to check if everything worked as expected"""
    tn = np.arange(0., domain.n_t * domain.dt, domain.dt)  # array with all timestamps
    xn = np.arange(domain.x_min, domain.x_max, domain.dx)  # array with all x_values

    fig = plt.figure(figsize=(5., 3.3))
    colorlist = [(0., 101 / 256., 189 / 256., 1.), (227/256., 114/256., 34/256., 1.)]

    for index, i in enumerate([0, domain.n_t-1]):  # plot IC and last time step
        subfig = fig.add_subplot(1, 1, 1)
        label = 't = ' + str(round(tn[i], 2))
        subfig.plot(xn, domain.grid[domain.i_ghost_l + 1:domain.i_ghost_r, i], label=label, color=colorlist[index])
        subfig.legend()

    plt.xlabel('$x$')
    plt.ylabel('$u(x, t)$')
    plt.title('Time evolution of solution')

    plt.savefig('transport-equation.png')
    plt.savefig('transport-equation.pgf')
