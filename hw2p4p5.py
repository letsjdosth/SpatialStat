from cmath import exp as c_exp
from math import exp, sin, tan, cos, pi

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from spatial_util.cov_functions import PoweredExp


def KL_approx_for_exp_cov(order, L_cov_domain_bound):
    def func_for_wj1(w):
        return tan(w*L_cov_domain_bound) - 1/w
    def func_for_wj2(w):
        return tan(w*L_cov_domain_bound) + w
    
    gap = pi/L_cov_domain_bound
    w1_array = []
    w2_array = []
    for i in range(order):
        root_inst1 = brentq(func_for_wj1, gap*i+0.00001, gap*(i+0.5)-0.000001)
        w1_array.append(root_inst1)
        root_inst2 = brentq(func_for_wj2, gap*(i+0.5)+0.00001, gap*(i+1)-0.00001)
        w2_array.append(root_inst2)

    # print(w1_array)
    # print(w2_array)

    def comp1_eigenval(w1, phi_range):
        return 2*phi_range/(w1**2 + phi_range**2)
    def comp1_eigenfunc(s, w1):
        return cos(w1*s)/(L_cov_domain_bound + sin(2*w1*L_cov_domain_bound)/(2*w1))**0.5

    def comp2_eigenval(w2, phi_range):
        return 2*phi_range/(w2**2 + phi_range**2)
    def comp2_eigenfunc(s, w2):
        return sin(w2*s)/(L_cov_domain_bound - sin(2*w2*L_cov_domain_bound)/(2*w2))**0.5

    s_grid = np.arange(0, L_cov_domain_bound, 0.02)
    comp = 0
    for w1, w2 in zip(w1_array, w2_array):
        comp += (comp1_eigenval(w1, 1)*np.array([comp1_eigenfunc(0, w1) for s in s_grid])*np.array([comp1_eigenfunc(s, w1) for s in s_grid]))
        comp += (comp2_eigenval(w2, 1)*np.array([comp2_eigenfunc(0, w2) for s in s_grid])*np.array([comp2_eigenfunc(s, w2) for s in s_grid]))
    return comp


L_cov_domain_bound = 5
s_grid = np.arange(0, L_cov_domain_bound, 0.02)
exp_inst = PoweredExp(1, 1, 1)
exp_inst.plot_covariance(0, L_cov_domain_bound, 0.02, show=False)
plt.plot(s_grid, KL_approx_for_exp_cov(1, L_cov_domain_bound))
plt.plot(s_grid, KL_approx_for_exp_cov(2, L_cov_domain_bound))
plt.plot(s_grid, KL_approx_for_exp_cov(4, L_cov_domain_bound))
plt.plot(s_grid, KL_approx_for_exp_cov(8, L_cov_domain_bound))
plt.plot(s_grid, KL_approx_for_exp_cov(16, L_cov_domain_bound))
plt.plot(s_grid, KL_approx_for_exp_cov(32, L_cov_domain_bound))
plt.show()


def fourier_basis_approx_for_exp_cov(order, L_cov_domain_bound):
    L = L_cov_domain_bound
    def eigenval(j):
        k = j*pi/(2*L)
        return 2*exp(-L) * (k*sin(k*L)-cos(k*L)+exp(L)) / (1+k**2)

    def eigenfunc(j, s):
        return c_exp(1j*j*pi*s/(2*L))/(2*L)**0.5
    
    s_grid = np.arange(0, L_cov_domain_bound, 0.02)
    
    comp = (0.5*eigenval(0)*np.array([eigenfunc(0, 0) for s in s_grid])*np.conjugate(np.array([eigenfunc(0, s) for s in s_grid])))
    for j in range(1,order+1):
        comp += (eigenval(j)*np.array([eigenfunc(j, 0) for s in s_grid])*np.conjugate(np.array([eigenfunc(j, s) for s in s_grid])))
    return comp

# print(fourier_basis_approx_for_exp_cov(1, L_cov_domain_bound))
exp_inst.plot_covariance(0, L_cov_domain_bound, 0.02, show=False)
plt.plot(s_grid, fourier_basis_approx_for_exp_cov(1, L_cov_domain_bound))
plt.plot(s_grid, fourier_basis_approx_for_exp_cov(2, L_cov_domain_bound))
plt.plot(s_grid, fourier_basis_approx_for_exp_cov(4, L_cov_domain_bound))
plt.plot(s_grid, fourier_basis_approx_for_exp_cov(8, L_cov_domain_bound))
plt.plot(s_grid, fourier_basis_approx_for_exp_cov(16, L_cov_domain_bound))
plt.plot(s_grid, fourier_basis_approx_for_exp_cov(32, L_cov_domain_bound))
plt.show()

