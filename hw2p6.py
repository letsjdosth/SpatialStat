from cmath import exp as c_exp
from cmath import sinh as c_sinh
from math import exp, sin, tan, cos, pi
from random import seed

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, brentq

from spatial_util.cov_functions import PoweredExp

exp_inst = PoweredExp(1,1,1)
n = 100
sample_path = exp_inst.sampler([np.array([i-n/2]) for i in range(n)])
# plt.plot([np.array([i-n/2]) for i in range(n)], sample_path)
# plt.show()

print("true\n", exp_inst.cov_matrix([np.array([i-n/2]) for i in range(n)]))
emp_cov = exp_inst.cov_matrix([np.array([i-n/2]) for i in range(n)]) #how?
print("est\n", emp_cov)

emp_eig_val, emp_eig_vec = np.linalg.eig(emp_cov)
print("emp:", emp_eig_val[0:5])
print(emp_eig_vec[:,0])


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

    def comp1_eigenval(w1, phi_range):
        return 2*phi_range/(w1**2 + phi_range**2)
    def comp1_eigenfunc(s, w1):
        return cos(w1*s)/(L_cov_domain_bound + sin(2*w1*L_cov_domain_bound)/(2*w1))**0.5

    def comp2_eigenval(w2, phi_range):
        return 2*phi_range/(w2**2 + phi_range**2)
    def comp2_eigenfunc(s, w2):
        return sin(w2*s)/(L_cov_domain_bound - sin(2*w2*L_cov_domain_bound)/(2*w2))**0.5

    eigenval1_array = [comp1_eigenval(w, 1) for w in w1_array]
    eigenval2_array = [comp2_eigenval(w, 1) for w in w2_array]
    return eigenval1_array, eigenval2_array

def fourier_basis_approx_for_exp_cov(order, L_cov_domain_bound):
    L = L_cov_domain_bound
    def eigenval(j):
        L = L_cov_domain_bound
        k = j*pi/(2*L_cov_domain_bound)
        return 2*exp(-L) * (k*sin(k*L)-cos(k*L)+exp(L)) / (1+k**2)

    def eigenfunc(j, s):
        return c_exp(1j*j*pi*s/(2*L))/(2*L)**0.5
    
    s_grid = np.arange(0, L_cov_domain_bound, 0.02)
    eigenval_array = []

    for j in range(1,order+1):
        eigenval_array.append(eigenval(j))
    return eigenval_array

KL_eig_val1, KL_eig_val2 = KL_approx_for_exp_cov(5, 50)
print("KL:", sorted(KL_eig_val1[0:5] + KL_eig_val2[0:5], reverse=True)[0:5])

fourier_eig_val = fourier_basis_approx_for_exp_cov(5, 50)
print("FB:",fourier_eig_val[0:5])

