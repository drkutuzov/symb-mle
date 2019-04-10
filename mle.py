###
from collections import namedtuple
from scipy.optimize import minimize
import numpy as np
import sympy


MLE_model = namedtuple('MLE_model', ['args', 'pars', 'L', 'LL', 'LL_jac', 'LL_hess'])


def make_MLE_model(symb_pdf):
    
    LL = sympy.log(symb_pdf.pdf)
    LL_jac = sympy.derive_by_array(LL, symb_pdf.pars)
    LL_hess = sympy.derive_by_array(LL_jac, symb_pdf.pars)
    
    return MLE_model(symb_pdf.args, symb_pdf.pars, symb_pdf.pdf, LL, LL_jac, LL_hess)


def lambdify_model(mle_model):
    
    args = [*mle_model.args, *mle_model.pars]
    
    return MLE_model(*mle_model[:2], *[sympy.lambdify(args, f) for f in mle_model[2:]])


def mle_model_iids(vals, mle_model):
    
    funcs = lambdify_model(mle_model)
    
    L = lambda p: np.prod(np.array([funcs.L(v, *p) for v in vals]), axis=0)
    LL = lambda p: np.sum(np.array([funcs.LL(v, *p) for v in vals]), axis=0)
    LL_jac = lambda p: np.sum(np.array([funcs.LL_jac(v, *p) for v in vals]), axis=0)
    LL_hess = lambda p: np.sum(np.array([funcs.LL_hess(v, *p) for v in vals]), axis=0)
    
    return MLE_model(*mle_model[:2], L, LL, LL_jac, LL_hess)


def mle_model_iids_hist(vals, counts, mle_model):
    
    funcs = lambdify_model(mle_model)
    
    L = lambda p: np.prod(np.array([funcs.L(v, *p)**n for v, n in zip(vals, counts)]), axis=0)
    LL = lambda p: np.sum(np.array([n*funcs.LL(v, *p) for v, n in zip(vals, counts)]), axis=0)
    LL_jac = lambda p: np.sum(np.array([n*np.array(funcs.LL_jac(v, *p)) for v, n in zip(vals, counts)]), axis=0)
    LL_hess = lambda p: np.sum(np.array([n*np.array(funcs.LL_hess(v, *p)) for v, n in zip(vals, counts)]), axis=0)
    
    return MLE_model(*mle_model[:2], L, LL, LL_jac, LL_hess)


def fit_mle(model, x0, method='bfgs', **kwg):
    
    def neg(f):
        return lambda *args: -f(*args)
    
    return minimize(neg(model.LL), x0, 
                        method=method,
                        jac=neg(model.LL_jac),
                        hess=neg(model.LL_hess),
                        **kwg)


def mle_param_covar(p_opt, model):
    fisher_info = -model.LL_hess(p_opt)
    covar = np.linalg.inv(fisher_info)
    return covar