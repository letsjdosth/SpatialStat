import csv
# from math import pi
# from functools import partial
# from random import seed, normalvariate

import numpy as np
import matplotlib.pyplot as plt

from spatial_util.least_squares import OLS_by_QR, GLS_by_cholesky
from spatial_util.cov_functions import Matern

from pyBayes.MCMC_Core import MCMC_Gibbs, MCMC_MH, MCMC_Diag

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



# full bayesian model

class FullBayes_UsingSocSd(MCMC_Gibbs):
    def __init__(self, initial, design_D, response_Z, nugget_sd, long_x, lat_y, hyper_sigma2S:tuple, hyper_phi:tuple, seed):
        self.MC_sample = [initial]
    
        self.design_D = np.array(design_D)
        self.response_Z = np.array(response_Z)
        self.location = [[x,y] for x,y in zip(long_x, lat_y)]
        self.N = len(response_Z)
        self.nugget_mat = np.diag(nugget_sd)**2

        self.hyper_sigma2S_alpha, self.hyper_sigma2S_beta = hyper_sigma2S
        self.hyper_phi_alpha, self.hyper_phi_beta = hyper_phi

        self.np_random_inst = np.random.default_rng(seed)


    def sampler(self, **kwargs):
        #  0     1         2    3
        # [beta, sigma2_S, phi, v]
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        #update new
        new = self.full_conditional_sampler_beta(new)
        new = self.full_conditional_sampler_sigma2_S(new)
        new = self.full_conditional_sampler_phi(new)
        new = self.full_conditional_sampler_v(new)
        self.MC_sample.append(new)

    def full_conditional_sampler_beta(self, last_param):
        #  0     1         2    3
        # [beta, sigma2_S, phi, v]
        new_sample = last_param
        #update new
        covS_inst = Matern(last_param[3], last_param[1], last_param[2])
        cov_mat = covS_inst.cov_matrix(self.location) + self.nugget_mat

        beta_fit, sum_of_squared_error, log_det_cov_mat_Sigma, F = GLS_by_cholesky(self.design_D, self.response_Z, cov_mat, return_F=True)

        nvm_mean = beta_fit
        nvm_cov = np.linalg.inv(np.transpose(F)@F)
        new_beta = self.np_random_inst.multivariate_normal(nvm_mean, nvm_cov)
        new_sample[0] = np.array(new_beta, dtype="float64")
        return new_sample

    def full_conditional_sampler_sigma2_S(self, last_param):
        #  0     1         2    3
        # [beta, sigma2_S, phi, v]
        new_sample = last_param
        #update new
        
        def log_post(sigma2_S):
            sigma2_S = sigma2_S[0]
            covS_inst = Matern(last_param[3], sigma2_S, last_param[2])
            cov_mat = covS_inst.cov_matrix(self.location) + self.nugget_mat
            
            fit_err = self.response_Z - self.design_D@last_param[0]
            _, logdet = np.linalg.slogdet(cov_mat)

            post_val = -0.5*logdet -0.5*np.transpose(fit_err)@np.linalg.inv(cov_mat)@fit_err
            post_val += ((-self.hyper_sigma2S_alpha-1)*np.log(sigma2_S) -self.hyper_sigma2S_beta/sigma2_S)
            return post_val

        def unif_proposal_log_pdf(from_smpl, to_smpl, window=1):
            from_smpl = from_smpl[0]
            to_smpl = to_smpl[0]
            applied_window = [max(0, from_smpl-window/2), from_smpl+window/2]
            if to_smpl<applied_window[0] or to_smpl>applied_window[1]:
                # return -inf
                raise ValueError("to_smpl has an unacceptable value")
            else:
                applied_window_len = applied_window[1] - applied_window[0]
                # return 1/applied_window_len
                return -np.log(applied_window_len)

        def unif_proposal_sampler(from_smpl, window=1):
            from_smpl = from_smpl[0]
            applied_window = [max(0, from_smpl-window/2), from_smpl+window/2]
            new = [self.np_random_inst.uniform(applied_window[0], applied_window[1])]
            return new

        mcmc_inst = MCMC_MH(log_post, unif_proposal_log_pdf, unif_proposal_sampler, [last_param[1]])
        mcmc_inst.generate_samples(2, verbose=False)
        sigma2_S = mcmc_inst.MC_sample[-1][0]
        new_sample[1] = sigma2_S
        return new_sample

    def full_conditional_sampler_phi(self, last_param):
        #  0     1         2    3
        # [beta, sigma2_S, phi, v]
        new_sample = last_param
        #update new
        
        def log_post(phi):
            phi = phi[0]
            covS_inst = Matern(last_param[3], last_param[1], phi)
            cov_mat = covS_inst.cov_matrix(self.location) + self.nugget_mat
            fit_err = self.response_Z - self.design_D@last_param[0]
            _, logdet = np.linalg.slogdet(cov_mat)

            post_val = -0.5*logdet -0.5*np.transpose(fit_err)@np.linalg.inv(cov_mat)@fit_err
            post_val += ((-self.hyper_phi_alpha-1)*np.log(phi)-self.hyper_phi_beta/phi)
            return post_val
        
        def unif_proposal_log_pdf(from_smpl, to_smpl, window=1):
            from_smpl = from_smpl[0]
            to_smpl = to_smpl[0]
            applied_window = [max(0, from_smpl-window/2), from_smpl+window/2]
            if to_smpl<applied_window[0] or to_smpl>applied_window[1]:
                # return -inf
                raise ValueError("to_smpl has an unacceptable value")
            else:
                applied_window_len = applied_window[1] - applied_window[0]
                # return 1/applied_window_len
                return -np.log(applied_window_len)

        def unif_proposal_sampler(from_smpl, window=1):
            from_smpl = from_smpl[0]
            applied_window = [max(0, from_smpl-window/2), from_smpl+window/2]
            new = [self.np_random_inst.uniform(applied_window[0], applied_window[1])]
            return new

        mcmc_inst = MCMC_MH(log_post, unif_proposal_log_pdf, unif_proposal_sampler, [last_param[2]])
        mcmc_inst.generate_samples(2, verbose=False)
        new_phi = mcmc_inst.MC_sample[-1][0]
        new_sample[2] = new_phi
        return new_sample
    
    def full_conditional_sampler_v(self, last_param):
        #  0     1         2    3
        # [beta, sigma2_S, phi, v]
        new_sample = last_param
        #update new
        
        def log_post(v):
            v = v[0]
            covS_inst = Matern(v, last_param[1], last_param[2])
            cov_mat = covS_inst.cov_matrix(self.location) + self.nugget_mat
            fit_err = self.response_Z - self.design_D@last_param[0]
            _, logdet = np.linalg.slogdet(cov_mat)

            post_val = -0.5*logdet -0.5*np.transpose(fit_err)@np.linalg.inv(cov_mat)@fit_err
            post_val += 0
            return post_val
        
        def choice_proposal_sampler(from_smpl):
            return [self.np_random_inst.choice([0.5, 1, 1.5, 2.5])]
        
        def symmetric_proposal_log_pdf(from_smpl, to_smpl):
            return 0
        
        mcmc_inst = MCMC_MH(log_post, symmetric_proposal_log_pdf, choice_proposal_sampler, [last_param[3]])
        mcmc_inst.generate_samples(2, verbose=False)
        new_v = mcmc_inst.MC_sample[-1][0]
        new_sample[3] = new_v
        return new_sample
    
    def full_conditional_sampler_fixed_v(self, last_param):
        new_sample = self.deep_copier(last_param)
        return new_sample
    
#  0     1         2    3
# [beta, sigma2_S, phi, v]
bayes_nugget_inst = FullBayes_UsingSocSd([np.array([0,0,0,0,0,0]), 1, 0.5, 1],
                                    design_mat_degree1_D1l, data_soil_carbon, data_soil_carbon_sd, data_long_x, data_lat_y,
                                    (0.01, 0.01), (0.01, 0.01), 20230223)
bayes_nugget_inst.generate_samples(2000, first_time_est=10, print_iter_cycle=20)

mcmc_diag_inst = MCMC_Diag()
samples_beta = [sample[0].tolist() for sample in bayes_nugget_inst.MC_sample]
samples_others = [sample[1:] for sample in bayes_nugget_inst.MC_sample]
samples_full = [x+y for x, y in zip(samples_beta, samples_others)]

mcmc_diag_inst.set_mc_samples_from_list(samples_full)
mcmc_diag_inst.write_samples("bayes_nugget_samples.csv")
#                                  0        1        2        3        4        5        6           7      8 
mcmc_diag_inst.set_variable_names(["beta0", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma2_S", "phi", "v"])
mcmc_diag_inst.show_traceplot((3,3), [0,1,2,3,4,5,6,7,8])
mcmc_diag_inst.show_hist((3,3))
