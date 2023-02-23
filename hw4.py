import csv
# from math import pi
# from functools import partial
# from random import seed, normalvariate

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as optim


from spatial_util.least_squares import OLS_by_QR, GLS_by_cholesky
from spatial_util.semi_variogram import semi_variogram, directional_semi_variogram, binning
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

# fig_eda, axs_eda = plt.subplots(1, 3, figsize=(15, 5))
# fig_eda.tight_layout()
# axs_eda0 = axs_eda[0].scatter(data_long_x, data_lat_y, s=data_soil_carbon, c=data_soil_carbon)
# axs_eda[0].set_title("soil carbon")
# axs_eda0_handles, axs_eda0_labels = axs_eda0.legend_elements(prop="colors", alpha=0.6)
# axs_eda[0].legend(axs_eda0_handles, axs_eda0_labels)
# axs_eda[1].scatter(data_long_x, data_lat_y, s=data_soil_carbon_sd, c=data_soil_carbon_sd)
# axs_eda[1].set_title("sd of soil carbon")
# axs_eda[2].scatter(data_long_x, data_lat_y, s=data_soil_carbon_sd, c=data_landuse_int)
# for i, txt in enumerate(data_landuse_str):
#     axs_eda[2].annotate(txt, (data_long_x[i], data_lat_y[i]))
# axs_eda[2].set_title("landuse")
# plt.show()


# trend fit
landuse_switcher_to_indicators = {'F':[0,0,0], 'W':[1,0,0], 'P':[0,1,0], 'X':[0,0,1]}
design_mat_degree1_D1n = np.array([[1, x, y] for x,y,_ in zip(data_long_x, data_lat_y, data_landuse_str)])
design_mat_degree1_D1l = np.array([[1, x, y] + landuse_switcher_to_indicators[z] for x,y,z in zip(data_long_x, data_lat_y, data_landuse_str)])
design_mat_degree1_D2n = np.array([[1, x, y, x**2, y**2, x*y] for x,y,_ in zip(data_long_x, data_lat_y, data_landuse_str)])
design_mat_degree1_D2l = np.array([[1, x, y, x**2, y**2, x*y] + landuse_switcher_to_indicators[z] for x,y,z in zip(data_long_x, data_lat_y, data_landuse_str)])

# trend_beta_D1n, _ = OLS_by_QR(design_mat_degree1_D1n, data_soil_carbon)
# trend_beta_D1l, _ = OLS_by_QR(design_mat_degree1_D1l, data_soil_carbon)
# trend_beta_D2n, _ = OLS_by_QR(design_mat_degree1_D2n, data_soil_carbon)
# trend_beta_D2l, _ = OLS_by_QR(design_mat_degree1_D2l, data_soil_carbon)

# trend_fit_D1n = design_mat_degree1_D1n @ trend_beta_D1n
# trend_fit_D1l = design_mat_degree1_D1l @ trend_beta_D1l
# trend_fit_D2n = design_mat_degree1_D2n @ trend_beta_D2n
# trend_fit_D2l = design_mat_degree1_D2l @ trend_beta_D2l

# trend_residual_D1n = data_soil_carbon - trend_fit_D1n
# trend_residual_D1l = data_soil_carbon - trend_fit_D1l
# trend_residual_D2n = data_soil_carbon - trend_fit_D2n
# trend_residual_D2l = data_soil_carbon - trend_fit_D2l

# print([round(b,3) for b in trend_beta_D1n])
# print([round(b,3) for b in trend_beta_D1l])
# print([round(b,3) for b in trend_beta_D2n])
# print([round(b,3) for b in trend_beta_D2l])

# fig_trend, axs_trend = plt.subplots(4, 5, figsize=(15, 12))
# fig_trend.tight_layout()
# axs_trend[0,0].set_title("data")
# axs_trend[0,1].set_title("fit")
# axs_trend[0,2].set_title("res")
# axs_trend[0,3].set_title("res vs fit")
# axs_trend[0,4].set_title("normalQQ for res")
# for i, (fit, res) in enumerate(zip([trend_fit_D1n, trend_fit_D1l, trend_fit_D2n, trend_fit_D2l],[trend_residual_D1n, trend_residual_D1l, trend_residual_D2n, trend_residual_D2l])):
#     sct0 = axs_trend[i,0].scatter(data_long_x, data_lat_y, s=data_soil_carbon, c=data_soil_carbon)
#     handles, labels = sct0.legend_elements(prop="colors", alpha=0.6)
#     axs_trend[i,0].legend(handles, labels)
#     sct1 = axs_trend[i,1].scatter(data_long_x, data_lat_y, s=fit, c=fit)
#     handles, labels = sct1.legend_elements(prop="colors", alpha=0.6)
#     axs_trend[i,1].legend(handles, labels)
#     axs_trend[i,2].scatter(data_long_x, data_lat_y, s=res, c=res)
#     axs_trend[i,3].scatter(fit, res)
#     axs_trend[i,3].axhline(0)
#     stats.probplot(res, dist=stats.norm, plot=axs_trend[i,4])
# plt.show()


# Choose trend_beta_D1l, plot variogram & fit

# dist_u, semi_varariogram_D1l = semi_variogram(trend_residual_D1l, data_long_x, data_lat_y)
# dist_u_d0, semi_varariogram_D1l_d0 = directional_semi_variogram(trend_residual_D1l, data_long_x, data_lat_y, 0) #right
# dist_u_d1, semi_varariogram_D1l_d1 = directional_semi_variogram(trend_residual_D1l, data_long_x, data_lat_y, 1) #left
# dist_u_d2, semi_varariogram_D1l_d2 = directional_semi_variogram(trend_residual_D1l, data_long_x, data_lat_y, 2) #up
# dist_u_d3, semi_varariogram_D1l_d3 = directional_semi_variogram(trend_residual_D1l, data_long_x, data_lat_y, 3) #down

# bin_dist_u, bin_vario_r = binning(dist_u, semi_varariogram_D1l, 30)
# bin_dist_u_d0, bin_vario_r_d0 = binning(dist_u_d0, semi_varariogram_D1l_d0, 12)
# bin_dist_u_d1, bin_vario_r_d1 = binning(dist_u_d1, semi_varariogram_D1l_d1, 12)
# bin_dist_u_d2, bin_vario_r_d2 = binning(dist_u_d2, semi_varariogram_D1l_d2, 12)
# bin_dist_u_d3, bin_vario_r_d3 = binning(dist_u_d3, semi_varariogram_D1l_d3, 12)

# print("semivariogram - the first bin:", round(bin_vario_r[0],4))
# fig_vario, axs_vario = plt.subplots(2, 5, figsize=(20, 8))
# fig_vario.tight_layout()
# axs_vario[0, 0].scatter(dist_u, semi_varariogram_D1l, s=0.5)
# axs_vario[0, 0].set_title("semivariogram cloud")
# axs_vario[0, 1].plot(bin_dist_u, bin_vario_r)
# axs_vario[0, 1].scatter(bin_dist_u, bin_vario_r)
# axs_vario[0, 1].set_title("semivariogram bin")
# axs_vario[0, 1].set_ylim(0, axs_vario[0, 0].get_ylim()[1])
# axs_vario[0, 2].plot(bin_dist_u, bin_vario_r)
# axs_vario[0, 2].scatter(bin_dist_u, bin_vario_r)
# axs_vario[0, 2].set_title("semivariogram bin(flexible axis)")
# axs_vario[0, 2].set_ylim(0, axs_vario[0, 2].get_ylim()[1])
# axs_vario[1, 0].scatter(dist_u_d0, semi_varariogram_D1l_d0, s=0.5)
# axs_vario[1, 0].set_title("semivariogram cloud, d0")
# axs_vario[1, 1].scatter(dist_u_d1, semi_varariogram_D1l_d1, s=0.5)
# axs_vario[1, 1].set_title("semivariogram cloud, d1")
# axs_vario[1, 2].scatter(dist_u_d2, semi_varariogram_D1l_d2, s=0.5)
# axs_vario[1, 2].set_title("semivariogram cloud, d2")
# axs_vario[1, 3].scatter(dist_u_d3, semi_varariogram_D1l_d3, s=0.5)
# axs_vario[1, 3].set_title("semivariogram cloud, d3")
# axs_vario[1, 4].plot(bin_dist_u_d0, bin_vario_r_d0)
# axs_vario[1, 4].plot(bin_dist_u_d1, bin_vario_r_d1)
# axs_vario[1, 4].plot(bin_dist_u_d2, bin_vario_r_d2)
# axs_vario[1, 4].plot(bin_dist_u_d3, bin_vario_r_d3)
# axs_vario[1, 4].set_title("semivariogram bin")
# axs_vario[1, 4].set_ylim(0, axs_vario[1, 4].get_ylim()[1])
# plt.show()

def minimize_func_base(x, nu_smoothness, bin_dist_u, bin_semivario_r):
    scale_sigma2 = x[0]
    range_phi = x[1]
    nugget_tau2 = x[2]

    bin_dist_u = bin_dist_u[0:-3]
    bin_semivario_r = bin_semivario_r[0:-3]

    
    matern_inst = Matern(nu_smoothness, scale_sigma2, range_phi)
    squared_sum = 0
    for u, r in zip(bin_dist_u, bin_semivario_r):
        squared_sum += ((matern_inst.semi_variogram(u) + nugget_tau2 - r)**2)
    return squared_sum

# minimize_func_v05 = partial(minimize_func_base, nu_smoothness=0.5, bin_dist_u=bin_dist_u, bin_semivario_r=bin_vario_r)
# optim_result_v05 = optim.minimize(minimize_func_v05, [100,1,90], method='nelder-mead', bounds=[(0,500), (0,10), (0, bin_vario_r[0])])
# # print("v0.5:",optim_result_v05)
# print("v0.5: sigma2=",optim_result_v05.x[0]," phi=",optim_result_v05.x[1]," tau2=",optim_result_v05.x[2])
# matern_inst_v05 = Matern(0.5, optim_result_v05.x[0], optim_result_v05.x[1])

# minimize_func_v10 = partial(minimize_func_base, nu_smoothness=1, bin_dist_u=bin_dist_u, bin_semivario_r=bin_vario_r)
# optim_result_v10 = optim.minimize(minimize_func_v10, [100,1,90], method='nelder-mead', bounds=[(0,500), (0,10), (0, bin_vario_r[0])])
# # print("v1.0:",optim_result_v10)
# print("v1.0: sigma2=",optim_result_v10.x[0]," phi=",optim_result_v10.x[1]," tau2=",optim_result_v10.x[2])
# matern_inst_v10 = Matern(1, optim_result_v10.x[0], optim_result_v10.x[1])


# minimize_func_v15 = partial(minimize_func_base, nu_smoothness=1.5, bin_dist_u=bin_dist_u, bin_semivario_r=bin_vario_r)
# optim_result_v15 = optim.minimize(minimize_func_v15, [100,1,90], method='nelder-mead', bounds=[(0,500), (0,10), (0, bin_vario_r[0])])
# # print("v1.5:",optim_result_v15)
# print("v1.5: sigma2=",optim_result_v15.x[0]," phi=",optim_result_v15.x[1]," tau2=",optim_result_v15.x[2])
# matern_inst_v15 = Matern(1.5, optim_result_v15.x[0], optim_result_v15.x[1])

# minimize_func_v25 = partial(minimize_func_base, nu_smoothness=2.5, bin_dist_u=bin_dist_u, bin_semivario_r=bin_vario_r)
# optim_result_v25 = optim.minimize(minimize_func_v25, [100,1,90], method='nelder-mead', bounds=[(0,500), (0,10), (0, bin_vario_r[0])])
# # print("v2.5:",optim_result_v25)
# print("v2.5: sigma2=",optim_result_v25.x[0]," phi=",optim_result_v25.x[1]," tau2=",optim_result_v25.x[2])
# matern_inst_v25 = Matern(2.5, optim_result_v25.x[0], optim_result_v25.x[1])


# vario_grid = np.arange(0, 6, 0.1)
# fig_vario_vs, axs_vario_vs = plt.subplots(1, 4, figsize=(4*4, 4))
# fig_vario_vs.tight_layout()
# axs_vario_vs[0].scatter(bin_dist_u, bin_vario_r)
# axs_vario_vs[0].plot(vario_grid, [matern_inst_v05.semi_variogram(g)+optim_result_v05.x[2] for g in vario_grid])
# axs_vario_vs[0].set_title("v=0.5")
# axs_vario_vs[1].scatter(bin_dist_u, bin_vario_r)
# axs_vario_vs[1].plot(vario_grid, [matern_inst_v10.semi_variogram(g)+optim_result_v10.x[2] for g in vario_grid])
# axs_vario_vs[1].set_title("v=1")
# axs_vario_vs[2].scatter(bin_dist_u, bin_vario_r)
# axs_vario_vs[2].plot(vario_grid, [matern_inst_v15.semi_variogram(g)+optim_result_v15.x[2] for g in vario_grid])
# axs_vario_vs[2].set_title("v=1.5")
# axs_vario_vs[3].scatter(bin_dist_u, bin_vario_r)
# axs_vario_vs[3].plot(vario_grid, [matern_inst_v25.semi_variogram(g)+optim_result_v25.x[2] for g in vario_grid])
# axs_vario_vs[3].set_title("v=2.5")
# plt.show()


# full bayesian model

class FullBayes(MCMC_Gibbs):
    def __init__(self, initial, design_D, response_Z, long_x, lat_y, hyper_sigma2T:tuple, hyper_theta:tuple, hyper_phi:tuple, seed):
        self.MC_sample = [initial]
    
        self.design_D = np.array(design_D)
        self.response_Z = np.array(response_Z)
        self.location = [[x,y] for x,y in zip(long_x, lat_y)]
        self.N = len(response_Z)

        self.hyper_sigma2T_alpha, self.hyper_sigma2T_beta = hyper_sigma2T
        self.hyper_theta_alpha, self.hyper_theta_beta = hyper_theta
        self.hyper_phi_alpha, self.hyper_phi_beta = hyper_phi

        self.np_random_inst = np.random.default_rng(seed)


    def sampler(self, **kwargs):
        #  0     1         2      3    4
        # [beta, sigma2_T, theta, phi, v]
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        #update new
        new = self.full_conditional_sampler_beta(new)
        new = self.full_conditional_sampler_sigma2_T(new)
        new = self.full_conditional_sampler_theta(new)
        new = self.full_conditional_sampler_phi(new)
        new = self.full_conditional_sampler_v(new)
        self.MC_sample.append(new)

    def full_conditional_sampler_beta(self, last_param):
        #  0     1         2      3    4
        # [beta, sigma2_T, theta, phi, v]
        new_sample = last_param
        #update new
        corr_inst = Matern(last_param[4], 1, last_param[3])
        cov_mat = last_param[1] * (corr_inst.cov_matrix(self.location)*last_param[2] + (1-last_param[2])*np.eye(self.N))

        beta_fit, sum_of_squared_error, log_det_cov_mat_Sigma, F = GLS_by_cholesky(self.design_D, self.response_Z, cov_mat, return_F=True)

        nvm_mean = beta_fit
        nvm_cov = np.linalg.inv(np.transpose(F)@F)
        new_beta = self.np_random_inst.multivariate_normal(nvm_mean, nvm_cov)
        new_sample[0] = np.array(new_beta, dtype="float64")
        return new_sample

    def full_conditional_sampler_sigma2_T(self, last_param):
        #  0     1         2      3    4
        # [beta, sigma2_T, theta, phi, v]
        new_sample = last_param
        #update new
        corr_inst = Matern(last_param[4], 1, last_param[3])
        corr_mat = corr_inst.cov_matrix(self.location)*last_param[2] + (1-last_param[2])*np.eye(self.N)
        fit_err = self.response_Z - self.design_D@last_param[0]

        gamma_shape = self.hyper_sigma2T_alpha + self.N/2
        gamma_rate = self.hyper_sigma2T_beta + 0.5 * np.transpose(fit_err)@np.linalg.inv(corr_mat)@fit_err
        new_sigma2_T = 1/self.np_random_inst.gamma(gamma_shape, 1/gamma_rate)
        new_sample[1] = new_sigma2_T
        return new_sample


    def full_conditional_sampler_theta(self, last_param):
        #  0     1         2      3    4
        # [beta, sigma2_T, theta, phi, v]
        new_sample = last_param
        #update new
        
        def log_post(theta):
            theta = theta[0]
            corr_inst = Matern(last_param[4], 1, last_param[3])
            cov_mat = last_param[1] * (corr_inst.cov_matrix(self.location)*theta + (1-theta)*np.eye(self.N))
            _, logdet = np.linalg.slogdet(cov_mat)
            inv_cov_mat = np.linalg.inv(cov_mat)

            fit_err = self.response_Z - self.design_D@last_param[0]
            
            post_val = -0.5*logdet -0.5*np.transpose(fit_err)@inv_cov_mat@fit_err
            post_val += ((self.hyper_theta_alpha-1)*np.log(theta) + (self.hyper_theta_beta-1)*np.log(1-theta))
            return post_val

        def unif_proposal_log_pdf(from_smpl, to_smpl, window=0.1):
            from_smpl = from_smpl[0]
            to_smpl = to_smpl[0]
            applied_window = [max(0, from_smpl-window/2), min(1, from_smpl+window/2)]
            if to_smpl<applied_window[0] or to_smpl>applied_window[1]:
                # return -inf
                raise ValueError("to_smpl has an unacceptable value")
            else:
                applied_window_len = applied_window[1] - applied_window[0]
                # return 1/applied_window_len
                return -np.log(applied_window_len)

        def unif_proposal_sampler(from_smpl, window=0.05):
            from_smpl = from_smpl[0]
            applied_window = [max(0, from_smpl-window/2), min(1, from_smpl+window/2)]
            new = [self.np_random_inst.uniform(applied_window[0], applied_window[1])]
            return new

        mcmc_inst = MCMC_MH(log_post, unif_proposal_log_pdf, unif_proposal_sampler, [last_param[2]])
        mcmc_inst.generate_samples(2, verbose=False)
        new_theta = mcmc_inst.MC_sample[-1][0]
        new_sample[2] = new_theta
        return new_sample
    
    def full_conditional_sampler_phi(self, last_param):
        #  0     1         2      3    4
        # [beta, sigma2_T, theta, phi, v]
        new_sample = last_param
        #update new
        
        def log_post(phi):
            phi = phi[0]
            corr_inst = Matern(last_param[4], 1, phi)
            cov_mat = (corr_inst.cov_matrix(self.location)*last_param[2] + (1-last_param[2])*np.eye(self.N))*last_param[1]
            _, logdet = np.linalg.slogdet(cov_mat)
            inv_cov_mat = np.linalg.inv(cov_mat)

            fit_err = self.response_Z - self.design_D@last_param[0]
            
            post_val = -0.5*logdet -0.5*np.transpose(fit_err)@inv_cov_mat@fit_err
            post_val += (-self.hyper_phi_beta/phi -(self.hyper_phi_alpha+1)*np.log(phi))
            return post_val
        
        def normal_proposal_sampler(from_smpl, sd=0.01):
            new = self.np_random_inst.normal(from_smpl, sd)
            return new
        
        def symmetric_proposal_log_pdf(from_smpl, to_smpl):
            return 0
        
        mcmc_inst = MCMC_MH(log_post, symmetric_proposal_log_pdf, normal_proposal_sampler, [last_param[3]])
        mcmc_inst.generate_samples(2, verbose=False)
        new_phi = mcmc_inst.MC_sample[-1][0]
        new_sample[3] = new_phi
        return new_sample
    
    def full_conditional_sampler_v(self, last_param):
        #  0     1         2      3    4
        # [beta, sigma2_T, theta, phi, v]
        new_sample = last_param
        #update new
        
        def log_post(v):
            v = v[0]
            corr_inst = Matern(v, 1, last_param[3])
            cov_mat = last_param[1] * (corr_inst.cov_matrix(self.location)*last_param[2] + (1-last_param[2])*np.eye(self.N))
            _, logdet = np.linalg.slogdet(cov_mat)
            inv_cov_mat = np.linalg.inv(cov_mat)

            fit_err = self.response_Z - self.design_D@last_param[0]
            
            post_val = -0.5*logdet -0.5*np.transpose(fit_err)@inv_cov_mat@fit_err
            post_val += 0
            return post_val
        
        def choice_proposal_sampler(from_smpl):
            return [self.np_random_inst.choice([0.5, 1, 1.5, 2.5])]
        
        def symmetric_proposal_log_pdf(from_smpl, to_smpl):
            return 0
        
        mcmc_inst = MCMC_MH(log_post, symmetric_proposal_log_pdf, choice_proposal_sampler, [last_param[4]])
        mcmc_inst.generate_samples(2, verbose=False)
        new_v = mcmc_inst.MC_sample[-1][0]
        new_sample[4] = new_v
        return new_sample
    
    def full_conditional_sampler_fixed_v(self, last_param):
        new_sample = self.deep_copier(last_param)
        return new_sample
    
#  0     1         2      3    4
# [beta, sigma2_T, theta, phi, v]
fullbayes_inst = FullBayes([np.array([0,0,0,0,0,0]), 1, 0.5, 0.1, 1], design_mat_degree1_D1l, data_soil_carbon, data_long_x, data_lat_y,
                           (0.01, 0.01), (1,5), (0.01, 0.01), 20230223)
fullbayes_inst.generate_samples(2000, first_time_est=10, print_iter_cycle=20)

mcmc_diag_inst = MCMC_Diag()
samples_beta = [sample[0].tolist() for sample in fullbayes_inst.MC_sample]
samples_others = [sample[1:] for sample in fullbayes_inst.MC_sample]
samples_signal_nugget = [[sample[1]*sample[2], sample[1]*(1-sample[2])] for sample in fullbayes_inst.MC_sample]
samples_full = [x+y+z for x, y, z in zip(samples_beta, samples_others, samples_signal_nugget)]

mcmc_diag_inst.set_mc_samples_from_list(samples_full)
#                                  0        1        2        3        4        5        6           7        8      9    10          11
mcmc_diag_inst.set_variable_names(["beta0", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma2_T", "theta", "phi", "v", "sigma2_S", "tau2"])
mcmc_diag_inst.show_traceplot((6,2), [0,1,2,3,4,5,6,7,8,9])
mcmc_diag_inst.show_hist((4,3))
mcmc_diag_inst.write_samples("fullbayes_samples.csv")