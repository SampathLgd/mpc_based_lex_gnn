# poly_softmax.py
import numpy as np

def exp_poly(x, degree=3):
    # approximate exp(x) via truncated Taylor: 1 + x + x^2/2 + x^3/6 ...
    y = np.ones_like(x)
    fac = 1.0
    px = 1.0
    for k in range(1, degree+1):
        px = px * x
        fac *= k
        y = y + px / fac
    return y

def reciprocal_newton(y, iterations=3):
    # compute 1/y approximately using Newton: start with float reciprocal
    r = 1.0 / y  # in MPC you'd need an initial public approx
    for _ in range(iterations):
        r = r * (2.0 - y * r)
    return r

def softmax_approx(scores, degree=3):
    # scores: numpy array
    smax = scores.max(axis=0)
    stabilized = scores - smax
    ex = exp_poly(stabilized, degree=degree)
    denom = ex.sum(axis=0)
    return ex / denom
