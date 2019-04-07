"""
Derivation of the first MDE of the KdV equation with Zabusky and Kruska scheme.
"""
from sympy import *

# initialize sympy variables:
x, t, tau, h = symbols(['x', 't', 'tau', 'h'])
v = Function('v')
init_printing(use_unicode=True)

# Taylor expansions:
order = 10  # order of Taylor expansion
# Taylor Series around v(x,t) in x and t
v_t_plus_dt = sum(tau**i/factorial(i) * v(x, t).diff(t, i) for i in range(order))
v_t_minus_dt = sum((-tau)**i/factorial(i) * v(x, t).diff(t, i) for i in range(order))
v_x_minus_h = sum((-h)**i/factorial(i) * v(x, t).diff(x, i) for i in range(order))
v_x_plus_h = sum(h**i/factorial(i) * v(x, t).diff(x, i) for i in range(order))
v_x_minus_2h = sum((-2*h)**i/factorial(i) * v(x, t).diff(x, i) for i in range(order))
v_x_plus_2h = sum((2*h)**i/factorial(i) * v(x, t).diff(x, i) for i in range(order))

# build scheme from Taylor expanded terms and set it to 0; this yields the first MDE
scheme = (v_t_plus_dt - v_t_minus_dt) / (2 * tau) + (v_x_plus_h + v(x, t) + v_x_minus_h) * \
         (v_x_plus_h - v_x_minus_h) / h + (v_x_plus_2h - 2 * v_x_plus_h + 2 * v_x_minus_h - v_x_minus_2h) / (2 * h**3)
scheme_basis = simplify(scheme)

print('First MDE:\n')
pprint(expand(scheme))
