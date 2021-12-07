import numpy as np
import scipy as sp
from scipy import optimize, special
import sympy as sy

def gaussian_wave(x, a, mu, sigma):
    z = (x-mu)/sigma
    return a*np.exp(-.5*z**2)

def gumbel_wave(x, a, mu, sigma):
    # note: mu is the mode in gumbel (not median)
    z = (x-mu)/sigma
    return a*np.e*np.exp(-(z+np.exp(-z))) # + is right gumbel, - left gumbel.

def model(x, a_p, mu_p, sigma_p, a_r, mu_r, sigma_r, a_s, mu_s, sigma_s, a_t, mu_t, sigma_t):
    p_wave = gaussian_wave(x, a_p, mu_p, sigma_p)
    r_wave = gaussian_wave(x, a_r, mu_r, sigma_r)
    s_wave = gaussian_wave(x, a_s, mu_s, sigma_s)
    t_wave = gumbel_wave(x, a_t, mu_t, sigma_t)
    return p_wave + r_wave + s_wave + t_wave

def z_pos_J():
    return np.log(1.5-np.sqrt(5)/2)

def z_pos(peak_percent):
    # x, c = sy.symbols('x c')
    # s = sy.solve(x + sy.exp(-x) - c, x)[0]
    # print(s)
    c = 1. - np.log(peak_percent/100.)
    v = c + sp.special.lambertw(-np.exp(-c))
    return v.real

def transform_matrix():

    T = np.array(
        [   
            [-1., 1., 0., 5/2., -3., 0.],
            [0., 0., 0., 5., 0., 0.],
            [0., 2., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., -1.],
            [0., 1., -1., 0., 0., 2.],
            [0., -1., 1., 0., 3., 2.]
        ], dtype=np.float64
    )

    return np.linalg.inv(T)

def t_end():
    # build error between  gumbel and line.
    y1 = sy.Symbol('y1', real=True, positive=False)
    y2 = sy.Symbol('y2', real=True, positive=True)
    a_n = sy.Symbol('a_n', real=True, positive=True)
    z = sy.symbols('z', real=True, positive=True)

    g = a_n * sy.exp(-(z + sy.exp(-z)))
    f = y1 * z + y2

    err_poly = sy.Integral((g-f)**2, (z, z_pos(70), z_pos(30))).doit()

    # aplico cambio de variable.
    sigma = sy.Symbol('sigma', real=True, positive=True)
    mu = sy.Symbol('mu', real=True, positive=True)
    x1 = sy.Symbol('x1', real=True, positive=False)
    x2 = sy.Symbol('x2', real=True, positive=True)

    err_poly_tr = err_poly * sigma
    err_poly_tr = err_poly_tr.subs(y1, sigma*x1)
    err_poly_tr = err_poly_tr.subs(y2, x2 + mu*x1)

    # gradiente  a cero 
    err_p1_x1 = sy.Derivative(err_poly_tr, x1).doit()
    err_p1_x2 = sy.Derivative(err_poly_tr, x2).doit()
    x1_opt, x2_opt = sy.linsolve([err_p1_x1, err_p1_x2], (x1, x2)).args[0]

    # computo de Tend
    T_end = -x2_opt.as_poly()/x1_opt.as_poly()
    # T_end = T_end.subs(a_n, sy.E * a)
    T_end = T_end.simplify()


    return T_end

def poly_coeffs():

    sigma = sy.Symbol('sigma', real=True, positive=True)
    mu = sy.Symbol('mu', real=True, positive=True)
    
    # inputs
    QRS = sy.Symbol('QRS', real=True, positive=True)
    QRS_on = sy.Symbol('QRS_on', real=True, positive=True)
    QT = sy.Symbol('QT', real=True, positive=True)
    J = QRS + QRS_on

    # aplico inputs sin evaluar 
    T_end = t_end()
    T_end_sub = T_end.subs(mu, J - z_pos_J()*sigma)

    # despejo sigma y polinomio característico.
    eq = sy.solveset(T_end_sub - (QT + QRS_on), sigma , domain=sy.S.Reals).args[0].args[1].args[0]

    # tomo coeffs en función de los inputs
    coeffs = eq.as_poly(sigma).coeffs()

    f = sy.lambdify([QT, QRS, QRS_on], coeffs)
    g = sy.lambdify([mu, sigma], T_end)

    return f, g

def temporal_gaussian_params(RR, PR, P, QRS, fun=None):
    v = np.array([PR, P, RR, 0., 0., QRS])
    return np.dot(fun, v) # mu_p, mu_r, mu_s,  sigma_p, sigma_r, sigma_s

def temporal_gumbel_params(QT, QRS, QRS_on, fun=None):

    roots = np.roots(fun(QT, QRS, QRS_on))
    cond = np.logical_and(np.isreal(roots), roots > 0)
    sigma_t = roots[cond][0].real

    J = QRS + QRS_on
    mu_t = J - z_pos_J() * sigma_t

    return mu_t, sigma_t

def nonlinear_system(x, y = None, params=None): # a_p, a_r, a_s, a_t

    params[0] = x[0]
    params[3] = x[1]
    params[6] = x[2]
    params[9] = x[3]

    f1 = y[0] - model(params[1], *params)
    f2 = y[1] - model(params[4], *params)
    f3 = y[2] - model(params[7], *params)
    f4 = y[3] - model(params[10], *params)

    return [ f1, f2, f3, f4 ]

def amplitude_params(y, fun=None):
    return sp.optimize.broyden1(fun, y) # # a_p, a_r, a_s, a_t

def fiducial_points(params, fun=None):
    """Compute fiducials from the model parameters. Required the Tend sympy function."""
    fiducials = np.empty(9, np.float64)

    fiducials[0] = params[1] - 2.5 * params[2] # Pon
    fiducials[1] = params[1] # P
    fiducials[2] = params[1] + 2.5 * params[2] #Pend

    fiducials[3] = params[4] - 3. * params[5] # QRSon
    fiducials[4] = params[4] # R
    fiducials[5] = params[7] # S
    fiducials[6] = params[10] + z_pos_J() * params[11] # QRSend / J

    fiducials[7] = params[10] # T
    fiducials[8] = fun(params[10], params[11]) # Tend

    return fiducials