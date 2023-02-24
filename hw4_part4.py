import csv
# from math import pi
from functools import partial
# from random import seed, normalvariate

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.optimize as optim

from spatial_util.least_squares import OLS_by_QR, GLS_by_cholesky, sym_defpos_matrix_inversion_cholesky
from spatial_util.cov_functions import Matern

from pyBayes.MCMC_Core import MCMC_Gibbs, MCMC_MH, MCMC_Diag

from hw4_part2 import FullBayes


# Data loading and plotting
data_soil_carbon = []
data_soil_carbon_sd = []
data_landuse_str = []
data_long_x = []
data_lat_y = []

data_path = ['data/soil_carbon_MD.csv', 'data/soil_carbon_NJ.csv', 'data/soil_carbon_DE.csv', 'data/soil_carbon_PA.csv']
for path in data_path:
    with open(path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csv_reader)
        # 0 ,1          ,2       ,3    ,4       ,5             ,6           ,7              ,8        ,9          ,10    ,11   ,12
        # "","sample_id","rcapid","soc","soc_sd","soc_measured","sample_top","sample_bottom","texture","elevation","long","lat","landuse"
        for row in csv_reader:
            # print(row)
            data_soil_carbon.append(float(row[3]))
            data_soil_carbon_sd.append(float(row[4]))
            data_long_x.append(float(row[10]))
            data_lat_y.append(float(row[11]))
            data_landuse_str.append(str(row[12])[1])
landuse_switcher = {'F':0, 'W':1, 'P':2, 'X':3}
data_landuse_int = [landuse_switcher[x]*10 for x in data_landuse_str]

# trend fit
landuse_switcher_to_indicators = {'F':[0,0,0], 'W':[1,0,0], 'P':[0,1,0], 'X':[0,0,1]}
design_mat_degree1_D1n = np.array([[1, x, y] for x,y,_ in zip(data_long_x, data_lat_y, data_landuse_str)])
design_mat_degree1_D1l = np.array([[1, x, y] + landuse_switcher_to_indicators[z] for x,y,z in zip(data_long_x, data_lat_y, data_landuse_str)])
design_mat_degree1_D2n = np.array([[1, x, y, x**2, y**2, x*y] for x,y,_ in zip(data_long_x, data_lat_y, data_landuse_str)])
design_mat_degree1_D2l = np.array([[1, x, y, x**2, y**2, x*y] + landuse_switcher_to_indicators[z] for x,y,z in zip(data_long_x, data_lat_y, data_landuse_str)])


# marginal likelihood model

def marginal_likelihood(range_phi, smoothness_nu, trend_design_X, resp_Y, data_points, nugget2_ratio=0):
    # nugget2_ratio = tau2 / sigmaS2
    range_phi = range_phi[0]
    n_data = len(resp_Y)
    matern_scale1_inst = Matern(smoothness_nu, 1, range_phi)
    matern_cov = matern_scale1_inst.cov_matrix(data_points)
    cov_mat = matern_cov + nugget2_ratio*np.eye(matern_cov.shape[0])

    # cov_mat_Sigma = LLt
    L = np.linalg.cholesky(cov_mat)
    log_det_matern_cov = np.sum(np.log(np.diag(L)))*2
    
    Z = scipy.linalg.solve_triangular(L, resp_Y, lower=True) #Z = L^(-1)Y
    F = scipy.linalg.solve_triangular(L, trend_design_X, lower=True) #F = L^(-1)X
    #now, Z = F*beta + error(with cov=I)
    
    beta_fit, S2 = OLS_by_QR(F, Z)
    _, logdet_DtVinvD = sym_defpos_matrix_inversion_cholesky(np.transpose(F)@F)
    
    m_lik = -0.5*log_det_matern_cov -0.5*logdet_DtVinvD -0.5*(n_data-len(beta_fit))*np.log(S2)
    return m_lik


class FitWith_MAPphi(FullBayes):
    def full_conditional_sampler_phi(self, last_param):
        #  0     1         2      3    4
        # [beta, sigma2_T, theta, phi, v]
        new_sample = last_param
        #update new
        
        now_nugget2_ratio = (1-last_param[2])/last_param[2]
        now_marginal_likelihood = partial(marginal_likelihood,
                                    smoothness_nu=last_param[4],
                                    trend_design_X=self.design_D, 
                                    resp_Y=self.response_Z,
                                    data_points=self.location, 
                                    nugget2_ratio=now_nugget2_ratio)
        def neg_marginal_posterior_optim_object(phi):
            return (-1)*(now_marginal_likelihood(phi) -self.hyper_phi_beta/phi -(self.hyper_phi_alpha+1)*np.log(phi)) #prior

        optim_result = optim.minimize(neg_marginal_posterior_optim_object, last_param[3], method='nelder-mead', bounds=[(0,10)], options={"maxiter":2})
        # print(optim_result) #for test
        new_phi = optim_result.x[0]
        new_sample[3] = new_phi
        return new_sample

if __name__ == "__main__":
    #  0     1         2      3    4
    # [beta, sigma2_T, theta, phi, v]
    map_phi_inst = FitWith_MAPphi([np.array([0,0,0,0,0,0]), 100, 0.5, 0.01, 1], design_mat_degree1_D1l, data_soil_carbon, data_long_x, data_lat_y,
                            (0.01, 0.01), (1,5), (0.01, 0.01), 20230223)
    map_phi_inst.generate_samples(2000, first_time_est=10, print_iter_cycle=20)

    mcmc_diag_inst = MCMC_Diag()
    samples_beta = [sample[0].tolist() for sample in map_phi_inst.MC_sample]
    samples_others = [sample[1:] for sample in map_phi_inst.MC_sample]
    samples_signal_nugget = [[sample[1]*sample[2], sample[1]*(1-sample[2])] for sample in map_phi_inst.MC_sample]
    samples_full = [x+y+z for x, y, z in zip(samples_beta, samples_others, samples_signal_nugget)]

    mcmc_diag_inst.set_mc_samples_from_list(samples_full)
    mcmc_diag_inst.write_samples("hw4_map_phi_samples")
    #                                  0        1        2        3        4        5        6           7        8      9    10          11
    mcmc_diag_inst.set_variable_names(["beta0", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma2_T", "theta", "phi", "v", "sigma2_S", "tau2"])
    mcmc_diag_inst.show_traceplot((6,2), [0,1,2,3,4,5,6,7,8,9])
    mcmc_diag_inst.show_hist((4,3))
