import numpy as np
import scipy as sp
from scipy import optimize, special, integrate
import sympy as sy
import dill 
import os


def build_transforms():
    global T_gumbel
    global T_end

    dill.settings['recurse'] = True
    if not( os.path.exists('coeffs') and os.path.exists('tend')):
        print('wait 100 seconds aprox ... only for the first opening.')
        T_gumbel, T_end = poly_coeffs()
        dill.dump(T_gumbel, open("coeffs", "wb"))
        dill.dump(T_end, open("tend", "wb"))

    T_end = dill.load(open("tend", "rb"))
    T_gumbel = dill.load(open("coeffs", "rb"))

    return [T_gumbel, T_end]

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

def params2fiducials(params, fun=None):
    """Compute fiducials from the model parameters. Required the Tend sympy function."""
    fiducials = np.full(13, np.nan, np.float32)

    fiducials[0] = params[1] - 2.5 * params[2] # Pon
    fiducials[1] = params[1] # P
    fiducials[2] = params[1] + 2.5 * params[2] #Pend

    fiducials[3] = params[4] - 3. * params[5] # QRSon
    fiducials[5] = params[4] # R
    fiducials[6] = params[7] # S
    fiducials[7] = params[10] + z_pos_J() * params[11] # QRSend / J

    fiducials[10] = params[10] # T
    fiducials[11] = fun(params[10], params[11]) # Tend

    return fiducials

def fiducials2inputs(fiducials, window, ecg):
    # 'Pon', 'P', 'Pend', 'QRSon', 'Q', 'R', 'S', 'QRSend','res','Ton', 'T', 'Tend', 'res'

    inputs = np.empty(9, dtype=np.float32)
    inputs[0] = window # 2*(fiducials[5]-onset) # RR
    inputs[1] = fiducials[2] - fiducials[0] # P duration
    inputs[2] = fiducials[3]-fiducials[0]# PR
    inputs[3] = fiducials[7]-fiducials[3]# QRS
    inputs[4] = fiducials[11]-fiducials[3]# QT
    inputs[5:] = ecg[fiducials[[1, 5, 6, 10]]]# peaks

    return inputs

def inputs2params(inputs, transforms=[None, None], flat=False):

    params = np.empty((4, 3), dtype=np.float32)

    RR = inputs[0]
    P = inputs[1]
    PR = inputs[2]
    QRS = inputs[3]
    QT = inputs[4]
    peaks = inputs[5:]

    # PQRS temporal part
    p = temporal_gaussian_params(RR, PR, P, QRS, fun=transforms[0])
    params[:3, 1] = p[:3]
    params[:3, 2] = p[3:6]

    # T temporal part
    t_r = params[1,1]
    sigma_r = params[1, 2]
    QRS_on = t_r - 3*sigma_r
    p = temporal_gumbel_params(QT, QRS, QRS_on, fun=transforms[1])
    params[3, 1:] = p

    # PQRST amplitudes
    flatten_params = np.ravel(params, order='C').tolist()
    F = lambda x: nonlinear_system(x, y=peaks, params=flatten_params)
    p = amplitude_params(peaks, fun=F)
    params[:, 0] = p

    if flat:
        params = np.ravel(params, order='C')

    return params

def perturbed_data(means, sds, n=100):
    return np.random.normal(means, sds, size=(n, means.size))

def ecgsyn_wr(n_beats, mean_inputs, sd_inputs, remove_drift=False):
    """rr en ms. params en ms y uV"""
    T_gauss = transform_matrix()
    T_gumbel, _ = build_transforms()

    inputs = perturbed_data(mean_inputs, sd_inputs, n=n_beats)
    rr = inputs[:,0]
    rr_acum = np.cumsum(rr)

    params = np.empty((n_beats, 12), dtype=np.float32)
    for i in range(n_beats):
        params[i,:] = inputs2params(inputs[i,:], transforms=[T_gauss, T_gumbel], flat=True)

    time = np.arange(0, rr_acum[-1])
    rad = np.radians(-135)
    sol = sp.integrate.solve_ivp(
        mc_sharry,
        args = (rr, rr_acum, params),
        t_span = (time[0], time[-1]),
        y0 = [np.cos(rad), np.sin(rad), 0], 
        t_eval = time,
        method='Radau'
    )

    if remove_drift:
        z = sol['y'][2,:]
        drift = np.linspace(z[0], z[-1], num=time.size)
        z -= drift

    return sol['t'], sol['y']

def dxdt(t, v, rr):
    alpha = 1. - np.sqrt(v[0]**2 + v[1]**2)
    w = 2*np.pi/rr
    return alpha * v[0] - w * v[1]

def dydt(t, v, rr):
    alpha = 1. - np.sqrt(v[0]**2 + v[1]**2)
    w = 2*np.pi/rr
    return alpha * v[1] + w * v[0]

def dgadt(t, v, rr, a, mu, sigma):
    w = 2*np.pi/rr
    t = (np.pi + np.arctan2(v[1], v[0])) / w
    z = (t-mu)/sigma
    g = gaussian_wave(t, a, mu, sigma)
    return -1.*g*z/sigma

def dgudt(t, v, rr, a, mu, sigma):
    w = 2*np.pi/rr
    t = (np.pi + np.arctan2(v[1], v[0])) / w
    z = (t-mu)/sigma
    g = gumbel_wave(t, a, mu, sigma)
    return g*(np.exp(-z)-1.)/sigma

def dzdt(t, v, rr, params):
    f = dgadt(t, v, rr, *params[:3])     \
        + dgadt(t, v, rr, *params[3:6])  \
        + dgadt(t, v, rr, *params[6:9])  \
        + dgudt(t, v, rr, *params[9:])
    
    return f

def mc_sharry(t, v, rr, rr_acum, params):
    i = np.nonzero(t < rr_acum)[0][0] # beat selector
    F = [
        dxdt(t, v, rr[i]),
        dydt(t, v, rr[i]),
        dzdt(t, v, rr[i], params[i,:])
    ]
    return F