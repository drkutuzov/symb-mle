###
from collections import namedtuple
import sympy


SymbPDF = namedtuple('SymbPDF', ['args', 'pars', 'pdf'])


def normal():
    x, mu, s = sympy.symbols('x mu s', real=True)
    a = (x - mu)**2/(2*s**2)
    pdf = sympy.exp(-(x - mu)**2/(2*s**2))/(sympy.sqrt(2*sympy.pi)*s)
    return SymbPDF(args=[x], pars=[mu, s], pdf=pdf)


def expon():
    x, mu = sympy.symbols('x mu', real=True)
    pdf = sympy.exp(-x/mu)/mu
    return SymbPDF(args=[x], pars=[mu], pdf=pdf)


def poisson():
    k, mu = sympy.symbols('k mu', real=True)
    pdf = sympy.exp(-mu)*mu**k/sympy.functions.combinatorial.factorials.factorial(k)
    return SymbPDF(args=[k], pars=[mu], pdf=pdf)


def sum_two_normal():
    x, p, mu1, s1, mu2, s2 = sympy.symbols('x p mu1 s1 mu2 s2', real=True)
    pdf1 = p*sympy.exp(-(x - mu1)**2/(2*s1**2))/(sympy.sqrt(2*sympy.pi)*s1)
    pdf2 = (1-p)*sympy.exp(-(x - mu2)**2/(2*s2**2))/(sympy.sqrt(2*sympy.pi)*s2)
    return SymbPDF(args=[x], pars=[p, mu1, s1, mu2, s2], pdf=pdf1 + pdf2)


def sum_two_normal_same_var():
    x, p, mu1, mu2, s = sympy.symbols('x p mu1 mu2 s', real=True)
    pdf1 = p*sympy.exp(-(x - mu1)**2/(2*s**2))/(sympy.sqrt(2*sympy.pi)*s)
    pdf2 = (1-p)*sympy.exp(-(x - mu2)**2/(2*s**2))/(sympy.sqrt(2*sympy.pi)*s)
    return SymbPDF(args=[x], pars=[p, mu1, mu2, s], pdf=pdf1 + pdf2)