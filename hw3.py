import csv
from math import pi
from functools import partial
from random import normalvariate

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as optim
import scipy.linalg

from spatial_util.least_squares import OLS_by_QR, GLS_by_cholesky
from spatial_util.cov_functions import Matern

data_soc = []
data_long = []
data_lat = []
with open('data/VA_soil_carbon.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(csv_reader)
    for row in csv_reader:
        data_soc.append(float(row[2]))
        data_long.append(float(row[9]))
        data_lat.append(float(row[10]))

# plt.scatter(data_long, data_lat, s=data_soc, c=data_soc)
# plt.show()


design_1st_X = np.array([[1, x, y] for x,y in zip(data_long, data_lat)])
design_2nd_X = np.array([[1, x, y, x**2, y**2, x*y] for x,y in zip(data_long, data_lat)])
resp_Y = np.array(data_soc)

trend_1st_coeff_OLS_fit, _ = OLS_by_QR(design_1st_X, resp_Y)
trend_2nd_coeff_OLS_fit, _ = OLS_by_QR(design_2nd_X, resp_Y)

trend_1st_fit = design_1st_X@trend_1st_coeff_OLS_fit
trend_2nd_fit = design_2nd_X@trend_2nd_coeff_OLS_fit

trend_1st_residual = resp_Y - trend_1st_fit
trend_2nd_residual = resp_Y - trend_2nd_fit

# fig1, axs1 = plt.subplots(1, 5, figsize=(15, 3))
# fig1.tight_layout()
# axs1[0].scatter(data_long, data_lat, s=data_soc, c=data_soc)
# axs1[0].set_title("data")
# axs1[1].scatter(data_long, data_lat, s=trend_1st_fit, c=trend_1st_fit)
# axs1[1].set_title("fit")
# axs1[2].scatter(data_long, data_lat, s=trend_1st_residual, c=trend_1st_residual)
# axs1[2].set_title("res")
# axs1[3].scatter(trend_1st_fit, trend_1st_residual)
# axs1[3].axhline(0)
# axs1[3].set_title("res vs fit")
# stats.probplot(trend_1st_residual, dist=stats.norm, plot=axs1[4])
# axs1[4].set_title("normalQQ for res")
# plt.show()

# fig2, axs2 = plt.subplots(1, 5, figsize=(15, 3))
# fig2.tight_layout()
# axs2[0].scatter(data_long, data_lat, s=data_soc, c=data_soc)
# axs2[0].set_title("data")
# axs2[1].scatter(data_long, data_lat, s=trend_2nd_fit, c=trend_2nd_fit)
# axs2[1].set_title("fit")
# axs2[2].scatter(data_long, data_lat, s=trend_2nd_residual, c=trend_2nd_residual)
# axs2[2].set_title("res")
# axs2[3].scatter(trend_2nd_fit, trend_2nd_residual)
# axs2[3].axhline(0)
# axs2[3].set_title("res vs fit")
# stats.probplot(trend_2nd_residual, dist=stats.norm, plot=axs2[4])
# axs2[4].set_title("normalQQ for res")
# plt.show()


def semi_variogram(data, long, lat):
    dist_u = []
    vario_r = []
    for i, (dat_i, long_i, lat_i) in enumerate(zip(data, long, lat)):
        for j, (dat_j, long_j, lat_j) in enumerate(zip(data, long, lat)):
            if i>j:
                pass
            else:
                dist_ij = ((long_i-long_j)**2 + (lat_i-lat_j)**2)**0.5
                vario_ij = 0.5 * (dat_i-dat_j)**2
                dist_u.append(dist_ij)
                vario_r.append(vario_ij)
    return dist_u, vario_r

def directional_semi_variogram(data, long, lat, direction):
    dist_u = []
    vario_r = []
    for i, (dat_i, long_i, lat_i) in enumerate(zip(data, long, lat)):
        for j, (dat_j, long_j, lat_j) in enumerate(zip(data, long, lat)):
            if i>j:
                pass
            else:
                if direction == 0 and long_i<long_j: #right
                    dist_ij = ((long_i-long_j)**2 + (lat_i-lat_j)**2)**0.5
                    vario_ij = 0.5 * (dat_i-dat_j)**2
                    dist_u.append(dist_ij)
                    vario_r.append(vario_ij)
                elif direction == 1 and long_i>long_j: #left
                    dist_ij = ((long_i-long_j)**2 + (lat_i-lat_j)**2)**0.5
                    vario_ij = 0.5 * (dat_i-dat_j)**2
                    dist_u.append(dist_ij)
                    vario_r.append(vario_ij)
                elif direction == 2 and lat_i<lat_j: #up
                    dist_ij = ((long_i-long_j)**2 + (lat_i-lat_j)**2)**0.5
                    vario_ij = 0.5 * (dat_i-dat_j)**2
                    dist_u.append(dist_ij)
                    vario_r.append(vario_ij)
                elif direction == 3 and lat_i>lat_j: #down
                    dist_ij = ((long_i-long_j)**2 + (lat_i-lat_j)**2)**0.5
                    vario_ij = 0.5 * (dat_i-dat_j)**2
                    dist_u.append(dist_ij)
                    vario_r.append(vario_ij)
                    
    return dist_u, vario_r

def binning(dist, vario, num_bins):
    max_u = max(dist)
    bin_tres_u = [(i+1)*max_u/num_bins for i in range(num_bins)]
    bin_middle_u = [(i+0.5)*max_u/num_bins for i in range(num_bins)]
    bin_vario_list = [[] for _ in range(num_bins)]
    # print(bin_tres_u)

    for u, r in zip(dist,vario):
        add_flag = False
        for i, tres in enumerate(bin_tres_u):
            if u<tres:
                bin_vario_list[i].append(r)
                add_flag = True
                break
        if not add_flag:
            bin_vario_list[-1].append(r)
            
    bin_vario_r = [np.mean(x) for x in bin_vario_list]
    # print(bin_vario_r)
    return bin_middle_u, bin_vario_r


dist_u, vario_r = semi_variogram(trend_2nd_residual, data_long, data_lat)
dist_u_d0, vario_r_d0 = directional_semi_variogram(trend_2nd_residual, data_long, data_lat, 0) #right
dist_u_d1, vario_r_d1 = directional_semi_variogram(trend_2nd_residual, data_long, data_lat, 1) #left
dist_u_d2, vario_r_d2 = directional_semi_variogram(trend_2nd_residual, data_long, data_lat, 2) #up
dist_u_d3, vario_r_d3 = directional_semi_variogram(trend_2nd_residual, data_long, data_lat, 3) #down

bin_dist_u, bin_vario_r = binning(dist_u, vario_r, 12)
bin_dist_u_d0, bin_vario_r_d0 = binning(dist_u_d0, vario_r_d0, 12)
bin_dist_u_d1, bin_vario_r_d1 = binning(dist_u_d1, vario_r_d1, 12)
bin_dist_u_d2, bin_vario_r_d2 = binning(dist_u_d2, vario_r_d2, 12)
bin_dist_u_d3, bin_vario_r_d3 = binning(dist_u_d3, vario_r_d3, 12)
print("possible nugget:", bin_vario_r[0])

# fig_vario, axs_vario = plt.subplots(1, 2, figsize=(8, 4))
# fig_vario.tight_layout()
# axs_vario[0].scatter(dist_u, vario_r, s=0.5)
# axs_vario[0].set_title("semivariogram cloud")
# axs_vario[1].plot(bin_dist_u, bin_vario_r)
# axs_vario[1].scatter(bin_dist_u, bin_vario_r)
# axs_vario[1].set_title("semivariogram bin")
# axs_vario[1].set_ylim(0, axs_vario[1].get_ylim()[1])
# plt.show()

# fig_vario_d, axs_vario_d = plt.subplots(1, 5, figsize=(20, 4))
# fig_vario_d.tight_layout()
# axs_vario_d[0].scatter(dist_u_d0, vario_r_d0, s=0.5)
# axs_vario_d[0].set_title("semivariogram cloud, d0")
# axs_vario_d[1].scatter(dist_u_d1, vario_r_d1, s=0.5)
# axs_vario_d[1].set_title("semivariogram cloud, d1")
# axs_vario_d[2].scatter(dist_u_d2, vario_r_d2, s=0.5)
# axs_vario_d[2].set_title("semivariogram cloud, d2")
# axs_vario_d[3].scatter(dist_u_d3, vario_r_d3, s=0.5)
# axs_vario_d[3].set_title("semivariogram cloud, d3")
# axs_vario_d[4].plot(bin_dist_u_d0, bin_vario_r_d0)
# axs_vario_d[4].plot(bin_dist_u_d1, bin_vario_r_d1)
# axs_vario_d[4].plot(bin_dist_u_d2, bin_vario_r_d2)
# axs_vario_d[4].plot(bin_dist_u_d3, bin_vario_r_d3)
# axs_vario_d[4].set_title("semivariogram bin")
# axs_vario_d[4].set_ylim(0, axs_vario_d[4].get_ylim()[1])
# plt.show()


def minimize_func_base(x, nu_smoothness, bin_dist_u, bin_semivario_r):
    bin_dist_u = bin_dist_u[0:-1]
    bin_semivario_r = bin_semivario_r[0:-1]

    scale_sigma2 = x[0]
    range_phi = x[1]
    matern_inst = Matern(nu_smoothness, scale_sigma2, range_phi)
    squared_sum = 0
    for u,r in zip(bin_dist_u, bin_semivario_r):
        squared_sum += ((matern_inst.semi_variogram(u) - r)**2)
    return squared_sum

# minimize_func_v05 = partial(minimize_func_base, nu_smoothness=0.5, bin_dist_u=bin_dist_u, bin_semivario_r=bin_vario_r)
# optim_result_v05 = optim.minimize(minimize_func_v05, [1,1], method='nelder-mead')
# print("v05:",optim_result_v05)
# matern_inst_v05 = Matern(0.5, optim_result_v05.x[0], optim_result_v05.x[1])

# minimize_func_v10 = partial(minimize_func_base, nu_smoothness=1, bin_dist_u=bin_dist_u, bin_semivario_r=bin_vario_r)
# optim_result_v10 = optim.minimize(minimize_func_v10, [1,1], method='nelder-mead')
# print("v10:",optim_result_v10)
# matern_inst_v10 = Matern(1, optim_result_v10.x[0], optim_result_v10.x[1])


# minimize_func_v15 = partial(minimize_func_base, nu_smoothness=1.5, bin_dist_u=bin_dist_u, bin_semivario_r=bin_vario_r)
# optim_result_v15 = optim.minimize(minimize_func_v15, [1,1], method='nelder-mead')
# print("v15:",optim_result_v15)
# matern_inst_v15 = Matern(1.5, optim_result_v15.x[0], optim_result_v15.x[1])

# minimize_func_v25 = partial(minimize_func_base, nu_smoothness=2.5, bin_dist_u=bin_dist_u, bin_semivario_r=bin_vario_r)
# optim_result_v25 = optim.minimize(minimize_func_v25, [1,1], method='nelder-mead')
# print("v25:",optim_result_v25)
# matern_inst_v25 = Matern(2.5, optim_result_v25.x[0], optim_result_v25.x[1])

# fig_vario_vs, axs_vario_vs = plt.subplots(1, 4, figsize=(4*4, 4))
# fig_vario_vs.tight_layout()
# axs_vario_vs[0].scatter(bin_dist_u, bin_vario_r)
# axs_vario_vs[0].set_title("v=0.5")
# matern_inst_v05.plot_semi_variogram(0, axs_vario_vs[0].get_xlim()[1], 0.01, plt_axis=axs_vario_vs[0], show=False)
# axs_vario_vs[1].scatter(bin_dist_u, bin_vario_r)
# axs_vario_vs[1].set_title("v=1")
# matern_inst_v10.plot_semi_variogram(0, axs_vario_vs[1].get_xlim()[1], 0.01, plt_axis=axs_vario_vs[1], show=False)
# axs_vario_vs[2].scatter(bin_dist_u, bin_vario_r)
# axs_vario_vs[2].set_title("v=1.5")
# matern_inst_v15.plot_semi_variogram(0, axs_vario_vs[2].get_xlim()[1], 0.01, plt_axis=axs_vario_vs[2], show=False)
# axs_vario_vs[3].scatter(bin_dist_u, bin_vario_r)
# axs_vario_vs[3].set_title("v=2.5")
# matern_inst_v25.plot_semi_variogram(0, axs_vario_vs[3].get_xlim()[1], 0.01, plt_axis=axs_vario_vs[3], show=False)
# plt.show()



# Q4, Q5

#need to estimate mle, again


def negative_profile_likelihood(scale_sigma2_range_phi, smoothness_nu, trend_design_X, resp_Y, data_long, data_lat, nugget=0):
    #profile: beta
    scale_sigma2 = scale_sigma2_range_phi[0]
    range_phi = scale_sigma2_range_phi[1]
    matern_scale1_inst = Matern(smoothness_nu, 1, range_phi)
    data_points = [[lon+normalvariate(0,0.1), lat+normalvariate(0,0.1)] for lon, lat in zip(data_long, data_lat)]
    matern_cov = matern_scale1_inst.cov_matrix(data_points)
    matern_cov = scale_sigma2*matern_cov + nugget*np.eye(matern_cov.shape[0])

    _, sse_gls, log_det_matern_cov_mat = GLS_by_cholesky(trend_design_X, resp_Y, matern_cov)

    p_lik = -0.5*log_det_matern_cov_mat - 0.5*sse_gls
    return -p_lik

def negative_profile_likelihood_2(range_phi, smoothness_nu, trend_design_X, resp_Y, data_long, data_lat, nugget=0):
    #profile: beta, sigma2
    range_phi = range_phi[0]
    n_data = len(resp_Y)
    matern_scale1_inst = Matern(smoothness_nu, 1, range_phi)
    data_points = [[lon+normalvariate(0,0.1), lat+normalvariate(0,0.1)] for lon, lat in zip(data_long, data_lat)]
    matern_cov = matern_scale1_inst.cov_matrix(data_points)
    matern_cov = matern_cov + nugget*np.eye(matern_cov.shape[0])

    _, sse_gls, log_det_matern_cov_mat = GLS_by_cholesky(trend_design_X, resp_Y, matern_cov)
    p_sigma2 = sse_gls/n_data

    p_lik = -0.5*n_data*np.log(2*pi*p_sigma2) -0.5*log_det_matern_cov_mat
    return -p_lik

pmle_optim_object = partial(negative_profile_likelihood, trend_design_X=design_2nd_X, resp_Y=resp_Y, data_long=data_long, data_lat=data_lat, nugget=0)
# optim_result_pmle = optim.minimize(pmle_optim_object, [298-0, 0.169], args=(1), method='Nelder-Mead')
# print(optim_result_pmle)

pmle_optim_object2 = partial(negative_profile_likelihood_2, trend_design_X=design_2nd_X, resp_Y=resp_Y, data_long=data_long, data_lat=data_lat, nugget=0)
# optim_result_pmle2 = optim.minimize(pmle_optim_object2, [0.169], args=(1), method='Nelder-Mead')
# print(optim_result_pmle2)


def gen_contour_level_matrix_for_likelihood(meshgrid_sigma2, meshgrid_phi, nu_smoothness):
    val_on_grid = np.empty(meshgrid_sigma2.shape)
    for i in range(meshgrid_sigma2.shape[0]):
        for j in range(meshgrid_sigma2.shape[1]):
            sigma2 = meshgrid_sigma2[i,j]
            phi = meshgrid_phi[i,j]
            val = pmle_optim_object([sigma2, phi], nu_smoothness)
            print(i, j, val)
            val_on_grid[i,j] = val
    return val_on_grid


# grid_sigma2 = np.linspace(220, 320, 10)
# grid_phi = np.linspace(0.02, 0.20, 10)
# meshgrid_sigma2, meshgrid_phi = np.meshgrid(grid_sigma2, grid_phi)

# print(optim_result_v10.x) #[298.652533, 0.169885285]
# contour_level_mat = gen_contour_level_matrix_for_likelihood(meshgrid_sigma2, meshgrid_phi, nu_smoothness=1)
# plt.contour(grid_sigma2, grid_phi, contour_level_mat, levels=10)
# plt.scatter([298.652533], [0.169885285])
# plt.show()


def marginal_likelihood(range_phi, smoothness_nu, trend_design_X, resp_Y, data_long, data_lat, nugget=0):
    range_phi = range_phi[0]
    n_data = len(resp_Y)
    matern_scale1_inst = Matern(smoothness_nu, 1, range_phi)
    data_points = [[lon+normalvariate(0,0.1), lat+normalvariate(0,0.1)] for lon, lat in zip(data_long, data_lat)]
    matern_cov = matern_scale1_inst.cov_matrix(data_points)
    matern_cov = matern_cov + nugget*np.eye(matern_cov.shape[0])

    # cov_mat_Sigma = LLt
    L = np.linalg.cholesky(matern_cov)
    log_det_matern_cov = np.sum(np.log(np.diag(L)))*2
    
    Z = scipy.linalg.solve_triangular(L, resp_Y, lower=True) #Z = L^(-1)Y
    F = scipy.linalg.solve_triangular(L, trend_design_X, lower=True) #F = L^(-1)X
    #now, Z = F*beta + error(with cov=I)
    
    beta_fit, S2 = OLS_by_QR(F, Z)
    DtVinvD = np.transpose(F)@F

    m_lik = -0.5*log_det_matern_cov -0.5*np.linalg.slogdet(DtVinvD)[1] -0.5*(n_data-len(beta_fit))*np.log(S2)

    return m_lik


grid_phi = np.linspace(0.02, 0.20, 20)
mmle_optim_object = partial(marginal_likelihood, trend_design_X=design_2nd_X, resp_Y=resp_Y, data_long=data_long, data_lat=data_lat, nugget=0)

level_list_marginal = [mmle_optim_object([p], 0.5) for p in grid_phi]
plt.plot(grid_phi, level_list_marginal)
plt.show()

level_list_marginal = [mmle_optim_object([p], 1) for p in grid_phi]
plt.plot(grid_phi, level_list_marginal)
plt.show()

level_list_marginal = [mmle_optim_object([p], 1.5) for p in grid_phi]
plt.plot(grid_phi, level_list_marginal)
plt.show()

level_list_marginal = [mmle_optim_object([p], 2.5) for p in grid_phi]
plt.plot(grid_phi, level_list_marginal)
plt.show()
