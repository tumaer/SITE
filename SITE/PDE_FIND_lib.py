"""
This library contains all functions necessary from the PDE-FIND framework to build the linear system matrix.
The matrix can consist of multiplicative combinations of derivatives and higher order temporal derivatives, too.
A function for preconditioning of the matrix is given as well. For a detailed description of the system see the paper.
"""
import numpy as np
import findiff
import math


def build_linear_system_FD(u, dt, dx, D=3, P=3, order_combinations=None,
                           high_order_time_derivs=0, acc_space=8, acc_time=8):
    """
    Constructs a linear system to use for sparse regression later.
    it used finite difference stencils from the findiff package.

    :param u: Grid data from the Simulation
    :param dt: Time step width
    :param dx: Spatial step width
    :param D: highest derivative to be included in library
    :param P: highest polynomial order to multiply u with derivative basis functions
    :param order_combinations: cumulative orders to include in the library
    :param high_order_time_derivs: highest order time derivative to include
    :param acc_space: accuracy orders of finite difference stencils in space
    :param acc_time: accuracy orders of finite difference stencils in time
    :return: ut, Theta, rhs_description; FD approximation to u_t, the system matrix Theta and the description of terms
    """

    n_x, n_t = u.shape

    # calculate highest derivatives to approximate in space and time
    if order_combinations is not None: combimax = max(order_combinations) - 1
    else: combimax = 0
    maxorder_x = max(D, combimax)
    maxorder_t = max(1, high_order_time_derivs)
    # calculate data padding width to avoid the use of non-centered stencils increasing the noise from FD approximations
    offset_t = padding_util(order_max=maxorder_t, acc=acc_time)
    offset_x = padding_util(order_max=maxorder_x, acc=acc_space)
    # new data width after padding
    n_t_2 = n_t - 2 * offset_t
    n_x_2 = n_x - 2 * offset_x

    # Approximate first time derivative (enough if no higher order time derivatives are considered)
    ut = np.zeros((n_x_2, n_t_2), dtype=np.float64)
    for i in range(n_x_2):  # use slices of the data: all values over time for a fixed spacial position
        whole_derivative = ApplyFindiff(u[i + offset_x, :], dt, 1, acc_time)  # only for spacial positions not padded
        # pad data afterwards to sort out the non-centered approximations
        ut[i, :] = whole_derivative[offset_t: -offset_t]
    ut = np.reshape(ut, (n_x_2 * n_t_2, 1), order='F')  # shapes the approximation, such that a vector is obtained

    # build a list that contains all combinations to be appended; any single derivatives are appended later
    # this list is used to build all feature vectors from basic derivative approximations and u
    combinations_list = []
    if order_combinations is not None:
        for combine_order in order_combinations:
            # adds all candidate terms consisting of combinations of lower derivative terms, to reach derivative D
            combinations_list += findCombinations(combine_order)  # constructs all combinations, which sum to D

    # initializations:
    Theta = np.zeros((n_x_2 * n_t_2, (D + 1 + len(combinations_list)) * (P + 1)), dtype=np.float64)  # system matrix
    ux = np.zeros((n_x_2, n_t_2), dtype=np.float64)  # spacial derivative storage
    rhs_description = ['' for i in range((D + 1 + len(combinations_list)) * (P + 1))]
    u_d_list = []

    u2 = u[offset_x:- offset_x, offset_t:- offset_t]  # save u in order to calculate powers of u e.g. in u**2 * u_xx

    # approximate all derivatives to use later in building of matrix
    for d in range(D + 1):  # iterate each derivative order to be approximated
        # approximate each derivative order
        if d > 0:
            for i in range(n_t_2):  # for each point in time consider spatial data
                whole_derivative = ApplyFindiff(u[:, i + offset_t], dx, d, acc_space)  # give all data to findiff
                ux[:, i] = whole_derivative[offset_x:-offset_x]  # pad data afterwards
        else:
            ux = np.ones((n_x_2, n_t_2), dtype=np.float64)  # represents intercept (will be reshaped to vector later)

        # save all derivatives as building blocks
        u_d_list.append(ux.copy())  # derivatives of any order are basis functions, to be extended with potentials of u
        combinations_list.append(np.array([d]))  # add single derivatives as candidate functions

    # build matrix Theta:
    for i, combination in enumerate(combinations_list):  # iterate all target candidate functions
        ux = np.ones((n_x_2, n_t_2), dtype=np.float64)  # reinitialize current derivative
        for derivative_term in combination:  # combination contains the derivative orders that are to be combined
            #  multiply derivatives to build more complex combinatorical terms
            ux *= u_d_list[derivative_term]  # u_d_list is sorted by derivative: [0 1 2 3 ...]

        for p in range(P + 1):  # multiplies powers of u with current derivative combination
            Theta[:, i * (P + 1) + p] = np.reshape(np.multiply(ux, np.power(u2, p)), (n_x_2 * n_t_2), order='F')

            # append description of current term
            if p == 1:
                rhs_description[i * (P + 1) + p] = rhs_description[i * (P + 1) + p] + 'u'
            elif p > 1:
                rhs_description[i * (P + 1) + p] = rhs_description[i * (P + 1) + p] + 'u^' + str(p)
            for derivative_term in combination:
                d = derivative_term
                if d > 0: rhs_description[i * (P + 1) + p] = rhs_description[i * (P + 1) + p] + \
                                                             'u_{' + ''.join(['x' for _ in range(d)]) + '}'

    # calculate higher order time derivatives using findiff:
    for time_order in range(2, high_order_time_derivs + 1):
        ut_higher = np.zeros((n_x_2, n_t_2), dtype=np.float64)  # initialize
        for i in range(n_x_2):  # all time values for constant spatial position
            whole_derivative = ApplyFindiff(u[i + offset_x, :], dt, time_order, acc_time)
            ut_higher[i, :] = whole_derivative[offset_t: -offset_t]  # cut data afterwards
        ut_higher = np.reshape(ut_higher, (n_x_2 * n_t_2, 1), order='F')
        # append higher order time derivative to matrix and description:
        Theta = np.concatenate((Theta, ut_higher), axis=1)
        rhs_description.append('u_{' + ''.join(['t' for _ in range(time_order)]) + '}')

    return ut, Theta, rhs_description


def ApplyFindiff(u, dx, d, acc):
    """
    Takes dth derivative of data using the findiff module.

    :param u: 1D-data vector to be differentiated
    :param dx: Grid spacing;  assumes uniform spacing
    :param d: Order
    :param acc: Order of accuracy of FD scheme
    :return: Vector of the same dimensions as u containing the derivative approximations
    """
    n = u.size
    ux = np.zeros(n, dtype=np.float64)

    dd_dxd = findiff.FinDiff(0, dx, d, acc=acc)  # initialization FD-Operator, for axis 0, and order acc
    ux[:] = dd_dxd(u)  # apply operator
    return ux


def padding_util(order_max, acc):
    """Helper function returns number of nodes to pad at each boundary; defined by the width of FD stencils"""
    return int(math.ceil((acc + order_max)/2)+1)


def findCombinations(target):
    """
    Calculates all combinations of derivative orders, which sum up to the 'target' cumulative order.
    E.g. if 'target' == 3: returns a list [[1, 1, 1], [1, 2]] containing all admissable combinations.
    Uses the recursive helper function 'findCombinationsUtil'.

    :param target: target cumulative order
    :return: list containing all admissable combinations
    """

    array = np.zeros(target, dtype=np.int)
    save_list = []
    findCombinationsUtil(array, 0, target, target, save_list)
    save_list.__delitem__(-1)  # last one is highest derivative; will be included anyway in matrix; hence deleted here
    return save_list


def findCombinationsUtil(array, index, target, reducedNum, save_list):
    """Solves the additive combinations problem recursively; is initialized by 'findCombinations'"""
    if reducedNum < 0: return  # end condition
    if reducedNum == 0:
        save_list.append(array[:index].copy())  # combination found; add it to array
        return
    if index == 0: prev = 1
    else: prev = array[index - 1]  # previous number; is used to maintain increasing order
    for k in range(prev, target + 1):
        array[index] = k  # next element is k
        findCombinationsUtil(array, index + 1, target, reducedNum - k, save_list)


def precondition_problem(preconditioner, R_raw, u_t_raw):
    """
    Preconditions a linear system according to the method determined by 'preconditioner'.

    :param preconditioner: String defining precondition method
    :param R_raw: System matrix Theta in the raw version from PDE-FIND
    :param u_t_raw: FD approximation to u_t in the raw version from PDE-FIND
    :return: R, u_t, inverse_transform; preconditioned system, preconditioned response vector and inverse transform
             to calculate the weight vector corresponding to the original system
    """

    if preconditioner == 'precondition':  # apply puffer transform
        # singular value decomposition: R_raw = U * D * V`
        # default precondition matrix: F = U * D^-1 * U^T

        # build precondition matrix F
        U_svd, singular_value_vect, _ = np.linalg.svd(R_raw, full_matrices=False)
        D_inv = np.zeros((R_raw.shape[1], R_raw.shape[1]))  # D^-1
        for i, sing_val in enumerate(singular_value_vect):
            D_inv[i, i] = 1. / sing_val
        temp_mat = np.dot(U_svd, D_inv)  # intermediate
        F = np.dot(temp_mat, U_svd.transpose())  # F can consume a lot of memory of the system is too large

        # apply puffer transform
        u_t = np.dot(F, u_t_raw)
        R = np.dot(F, R_raw)
        inverse_transform = np.identity(R.shape[1])  # no backwards transform of weight vector necessary

    elif preconditioner == 'norm':  # Scale the system
        # R_raw * D^-1 * D * x = u_t_raw
        X_T_X = np.dot(R_raw.transpose(), R_raw)
        D_inv = np.zeros((R_raw.shape[1], R_raw.shape[1]))
        for i in range(D_inv.shape[0]):  # default choice of D_inv
            D_inv[i, i] = 1. / np.sqrt(X_T_X[i, i])

        # scale system
        R = np.dot(R_raw, D_inv)
        u_t = u_t_raw  # no change in u_t necessary
        inverse_transform = D_inv  # in order to transform weight vector back

    elif preconditioner == 'norm_precondition':  # norm first, then apply puffer transform
        R_norm, _, inverse_transform = precondition_problem('norm', R_raw, u_t_raw)  # u_t is unchanged when norming
        R, u_t, _ = precondition_problem('precondition', R_norm, u_t_raw)  # inverse transform of puffer is Identity

    else:  # use raw data
        R = R_raw
        u_t = u_t_raw
        inverse_transform = np.identity(R.shape[1])  # no backwards transform of weight vector necessary

    return R, u_t, inverse_transform


def print_pde(w, rhs_description):
    """
    Prints the predicted MDE in readable form

    :param w: predicted weight vector
    :param rhs_description: Description corresponding to 'w'
    :return: None
    """

    pde = 'u_t  = '
    first = True
    for i in range(len(w)):
        if w[i] != 0:  # term is in equation
            if not first:
                pde = pde + ' + '
            pde = pde + str(w[i]) + rhs_description[i] + "\n   "
            first = False
    print(pde)
