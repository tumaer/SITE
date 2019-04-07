"""
Derivation of the third MDE of the advection equation with FTBS discretization up to 6th order.
We apply the procedure of Warming and Hyett.
"""
from sympy import *

# initialize sympy variables:
x, t, tau, h, a = symbols(['x', 't', 'tau', 'h', 'a'])
v = Function('v')
init_printing(use_unicode=True)

# Taylor expansions:
order = 8  # order of Taylor expansion; needs to be larger than the highest time derivative to be eliminated
# Taylor expansion of v(x,t + dt) about v(x, t)
v_t_plus_dt = sum(tau**i/factorial(i) * v(x, t).diff(t, i) for i in range(order))
v_Delta_t = (v_t_plus_dt - v(x, t))/tau  # forward-time approximation of FTBS using Taylor series of v_t_plus_dt
v_Delta_t = simplify(v_Delta_t)

# Taylor expansion of v(x - dx, t) about v(x, t)
v_x_minus_h = sum((-h)**i/factorial(i) * v(x, t).diff(x, i) for i in range(order))
v_Delta_x = a * (v(x, t) - v_x_minus_h) / h  # backwards-space approximation of FTBS using Taylor series of v_x_minus_h
v_Delta_x = simplify(v_Delta_x)

# set Taylor expanded scheme to 0; this is the first MDE
FTBS = v_Delta_t + v_Delta_x
FTBS_basis = simplify(FTBS)
print('First MDE:\n')
pprint(expand(FTBS_basis))

# Apply the procedure of Warming and Hyett to eliminate temporal derivatives for spatial derivatives
# Eliminate 2nd temporal derivative:
FTBS = FTBS + FTBS_basis.diff(t) * (-tau/2)
FTBS = FTBS + FTBS_basis.diff(x) * (a*tau/2)

# Eliminate 3rd temporal derivative:
FTBS = FTBS + FTBS_basis.diff(t, t) * (tau**2)/12
FTBS = FTBS + FTBS_basis.diff(x, t) * ((-a * tau**2)/3)
FTBS = FTBS + FTBS_basis.diff(x, x) * (-a * tau*h/4 + a**2*tau**2/3)

# Eliminate 4th temporal derivative:
FTBS = FTBS + FTBS_basis.diff(x, t, t) * (a*tau**3/12)
FTBS = FTBS + FTBS_basis.diff(x, x, t) * (a*h*tau**2/6 - a**2*tau**3/4)
FTBS = FTBS + FTBS_basis.diff(x, x, x) * (a*h**2*tau/12 - a**2*tau**2*h/3 + a**3*tau**3/4)

# Eliminate 5th temporal derivative:
FTBS = FTBS + FTBS_basis.diff(t, t, t, t) * (-tau**4/720)
FTBS = FTBS + FTBS_basis.diff(x, t, t, t) * (-a*tau**4/180)
FTBS = FTBS + FTBS_basis.diff(x, x, t, t) * (-a*h*tau**3/24 + 3*a**2*tau**4/40)
FTBS = FTBS + FTBS_basis.diff(x, x, x, t) * (-a*h**2*tau**2/18 + a**2*h*tau**3/4 - a**3*tau**4/5)
FTBS = FTBS + FTBS_basis.diff(x, x, x, x) * (-a*h**3*tau/48 + 7*a**2*h**2*tau**2/36 - 3*a**3*h*tau**3/8 + a**4*tau**4/5)

# Eliminate 6th temporal derivative:
FTBS = FTBS + FTBS_basis.diff(x, t, t, t, t) * (-a*tau**5/720)
FTBS = FTBS + FTBS_basis.diff(x, x, t, t, t) * (a*h*tau**4/360 - a**2*tau**5/120)
FTBS = FTBS + FTBS_basis.diff(x, x, x, t, t) * (a*h**2*tau**3/72 - 3*a**2*h*tau**4/40 + a**3*tau**5/15)
FTBS = FTBS + FTBS_basis.diff(x, x, x, x, t) * (a*h**3*tau**2/72 - 7*a**2*h**2*tau**3/48 + 3*a**3*h*tau**4/10 -
                                            a**4*tau**5/6)
FTBS = FTBS + FTBS_basis.diff(x, x, x, x, x) * (a*h**4*tau/240 - a**2*h**3*tau**2/12 + 5*a**3*h**2*tau**3/16 -
                                            2*a**4*h*tau**4/5 + a**5*tau**5/6)
print('Third MDE:\n')
pprint(expand(FTBS))


# Resulting third MDE up to 6th order:
# 0 = v_t + a v_x + (-a*h/2 + a**2*h/2  )*v_xx + (a*h**2/6 - a**2*h*tau/2 + a**3*tau**2/3)*v_xxx
#   + (-a*h**3/24 + a**2*h**2*tau*7/24 - a**3*h*tau**2/2 + a**4*tau**3/4) v_xxxx
#   + (a*h**4/120 - a**2*h**3*tau/8 + a**3*h**2*tau**2*5/12 - a**4*h*tau**3/2 + a**5*tau**4/5) v_xxxxx
#  + (-a*h**5/720 + 31*a**2*h**4*tau/720 - a**3*h**3*tau**2/4 + 13*a**4*h**2*tau**3/24 - a**5*h*tau**4/2 +
#      a**6*tau**5/6) v_xxxxxx
