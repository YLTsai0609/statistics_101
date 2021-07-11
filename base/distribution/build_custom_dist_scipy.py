# -*- coding: utf-8 -*-
'''
還是使用scipy來fit distribution而不是numpy，因為可以直接利用scipy fit裡面的
parameter solver來實作MLE(Maximum Likelihood Estimation) 
- you cloud also do max log likelihood or min negitive log likelihood


1. Fitting empirical distribution to theoretical ones with Scipy (Python)? 
- Fit scipy裡面全部內建的81種dist，找到一個fit的最好的，從所有dist中挑出
Sum of Square Error (SSE)最小的，尚未經過KS test
https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

2. Fitting data with a custom distribution using scipy.stats
- Fit Skewed generalized t distribution 自定義的，from 實際觀察Data
https://stackoverflow.com/questions/52207964/fitting-data-with-a-custom-distribution-using-scipy-stats

3. implementation
3.1 - possion proccess -> choose gamma distribution(alpha, beta) - the conjugate prior for the parameter in a Poisson process.

gamma distribution(alpha, beta) on wiki https://en.wikipedia.org/wiki/Gamma_distribution
f_gamma(x; alpha, beta) = beta ** alpha * x (alpha - 1) exp** (- beta * x) / gamma(alpha), where gamma(alpha) = (alpha - 1) !

3.2 seasonality f_sin(t : rho, omega, phi) = rho * sin(omega * t + phi)

3.3 our analytical hypothesis : f(t : alpha, beta, rho, omega, phi) = f_gamma * (1 + f_sin)

RECORD
0301 實作上有一些問題，為了建立一個distribution物件，使用scipy.stats.rv_continuous這個subclass來建造
但是問題很多，包含產生random variable時，想必fit時也會有很多問題
以下產生random variable時的問題
RuntimeWarning : overflow when np.exp - 應該可解
IntegrationWarning : The integral is probably divergnet, or slowly cconvergent - integrae.quad
IntegrationWarning : The maximum nunber of subdivisions(50) has been achieved If increasing the limit yields no improvement it is advised to analyze 
  the integrand in order to determine the difficulties.  If the position of a 
  local difficulty can be determined (singularity, discontinuity) one will 
  probably gain from splitting up the interval and calling the integrator 
  on the subranges.  Perhaps a special-purpose integrator should be used.
  return integrate.quad(self._pdf, _a, x, args=args)[0]
IntegrationWarning: The algorithm does not converge.  Roundoff error is detected in the extrapolation table.  It is assumed that the requested tolerance
  cannot be achieved, and that the returned result (if full_output = 1) is the best which can be obtained.
  return integrate.quad(self._pdf, _a, x, args=args)[0]

RuntimeWarning: overflow encountered in ? (vectorized)
  outputs = ufunc(*inputs)

RuntimeWarning: divide by zero encountered in double_scalars
  Shat = sqrt(mu2hat / mu2)

RuntimeWarning: invalid value encountered in subtract
  numpy.max(numpy.abs(fsim[0] - fsim[1:])) <= fatol):

辦法 : 
1. 查看其他scipy有建立的distribution如何解決此問題? Check Gamma distribution
2. 繞道而行，直接使用ML approach
3. 使用numpy手刻一個optimizer，輸入likilihood function - 其實沒有解決問題
4. find a kde solution

'''


import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import gamma, beta
# from math import exp, gamma, sin

class sgt(st.rv_continuous):

    def _pdf(self, x, mu, sigma, lam, p, q):

        v = q ** (-1 / p) * \
            ((3 * lam ** 2 + 1) * (
                    beta(3 / p, q - 2 / p) / beta(1 / p, q)) - 4 * lam ** 2 *
             (beta(2 / p, q - 1 / p) / beta(1 / p, q)) ** 2) ** (-1 / 2)

        m = 2 * v * sigma * lam * q ** (1 / p) * beta(2 / p, q - 1 / p) / beta(
            1 / p, q)

        fx = p / (2 * v * sigma * q ** (1 / p) * beta(1 / p, q) * (
                abs(x - mu + m) ** p / (q * (v * sigma) ** p) * (
                lam * np.sign(x - mu + m) + 1) ** p + 1) ** (
                          1 / p + q))

        return fx

    def _argcheck(self, mu, sigma, lam, p, q):

        s = sigma > 0
        l = -1 < lam < 1
        p_bool = p > 0
        q_bool = q > 0

        all_bool = s & l & p_bool & q_bool

        return all_bool

# 可以work
# sgt_inst = sgt(name='sgt')
# vars = sgt_inst.rvs(mu=1, sigma=3, lam = -0.1, p = 2, q = 50, size = 100)
# args = sgt_inst.fit(vars, 0.5, 0.5, -0.5, 2, 10)

# print(args)

class demand_curve(st.rv_continuous):
    '''
    Gamma function \times seasonality pattern (cos fumction)
    '''
    def _pdf(self, x, alpha, beta, rho, omega, phi):
        f_gamma = (beta ** alpha) * (x ** (alpha - 1)) * np.exp(-beta * x) / gamma(alpha)
        if f_gamma == np.inf:
            f_gamma = np.exp(500)
        elif f_gamma == -np.inf:
            f_gamma =  - np.exp(500)

        f_sin = rho * np.sin(omega * x + phi)
        f = f_gamma * (1 + f_sin)
        # f = f_sin
        return f
    def _argcheck(self, alpha, beta, rho, omega, phi):
        alpha_postive = alpha > 0
        beta_postive = beta > 0
        rho_positve = rho > 0
        omega_positive = omega > 0

        all_bool = alpha_postive & beta_postive & rho_positve & omega_positive

        return all_bool

curve = demand_curve('listA')
rv = curve.rvs(alpha=1, beta=0.03, rho=1, omega=2, phi=3, size=100)
args = curve.fit(rv, 0.5, 0.5, -0.5, 2, 10)

print(args)
