import csv
from pyBayes.MCMC_Core import MCMC_Diag
from spatial_util.cov_functions import Matern
from spatial_util.least_squares import sym_defpos_matrix_inversion_cholesky

import numpy as np
import matplotlib.pyplot as plt

part2_inst = MCMC_Diag()
part2_MC_sample = []
with open("hw4_fullbayes_samples_t1.csv", newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csv_reader:
        part2_MC_sample.append([float(x) for x in row])
with open("hw4_fullbayes_samples_t2.csv", newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csv_reader:
        part2_MC_sample.append([float(x) for x in row])
part2_inst.set_mc_samples_from_list(part2_MC_sample)
part2_inst.burnin(100)
#                              0        1        2        3        4        5        6           7        8      9    10          11
part2_inst.set_variable_names(["beta0", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma2_T", "theta", "phi", "v", "sigma2_S", "tau2"])
# part2_inst.show_traceplot((6,2))
# part2_inst.show_hist((6,2))
# part2_inst.show_acf(30, (6,2))
# part2_inst.print_summaries(4, latex_table_format=True)


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
# landuse_switcher = {'F':0, 'W':1, 'P':2, 'X':3}

data_pts = [(x,y) for x ,y in zip(data_long_x, data_lat_y)]
landuse_switcher_to_indicators = {'F':[0,0,0], 'W':[1,0,0], 'P':[0,1,0], 'X':[0,0,1]}
design_mat_degree1_D1l = np.array([[1, x, y] + landuse_switcher_to_indicators[z] for x,y,z in zip(data_long_x, data_lat_y, data_landuse_str)])

# ============================================================
plt_cm = plt.cm.get_cmap()
fig_eda, axs_eda = plt.subplots(1, 2, figsize=(10, 5))
fig_eda.tight_layout()
axs_eda0 = axs_eda[0].scatter(data_long_x, data_lat_y, c=data_soil_carbon, vmin=0, vmax=55, cmap=plt_cm)
axs_eda[0].set_title("soil carbon and landuse")
for i, txt in enumerate(data_landuse_str):
    axs_eda[0].annotate(txt, (data_long_x[i], data_lat_y[i]))
axs_eda0_handles, axs_eda0_labels = axs_eda0.legend_elements(prop="colors", alpha=0.6)
axs_eda[0].legend(axs_eda0_handles, axs_eda0_labels)

x_knots = np.arange(axs_eda[0].get_xlim()[0], axs_eda[0].get_xlim()[1], 0.1) #<<<sensitive!
y_knots = np.arange(axs_eda[0].get_ylim()[0], axs_eda[0].get_ylim()[1], 0.1) #<<<sensitive!
grid_pts = [(x,y) for x in x_knots for y in y_knots]
axs_eda[0].scatter(*zip(*grid_pts), s=0.5)
# plt.show()
krigging_design_mat = np.array([[1, x, y] + [0,0,0] for (x, y) in grid_pts]) #all F
for i, ((x, y), lu) in enumerate(zip(data_pts, data_landuse_str)):
    radius = 0.15 #<<<sensitive!
    if lu != 'F':
        for row_idx, row in enumerate(krigging_design_mat):
            dist = ((x-row[1])**2+(y-row[2])**2)**0.5
            if dist<radius:
                krigging_design_mat[row_idx, 3:6] = np.array(landuse_switcher_to_indicators[lu])

# ============================================================
# krigging
# #                              0        1        2        3        4        5        6           7        8      9    10          11
# part2_inst.set_variable_names(["beta0", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma2_T", "theta", "phi", "v", "sigma2_S", "tau2"])
n_data_pts = len(data_pts)
n_grid_pts = len(grid_pts)
print(n_data_pts, n_grid_pts, n_data_pts+n_grid_pts)
part2_sample_mean = part2_inst.get_sample_mean()
matern_inst = Matern(part2_sample_mean[9], part2_sample_mean[10],part2_sample_mean[8])
krigging_cov_mat = matern_inst.cov_matrix(grid_pts+data_pts) #grid first
print("cov-mat is constructed")
cov_grid_grid = krigging_cov_mat[0:n_grid_pts, 0:n_grid_pts]
cov_data_data = krigging_cov_mat[n_grid_pts:, n_grid_pts:] + np.diag([part2_sample_mean[11] for _ in range(n_data_pts)])
cov_grid_data = krigging_cov_mat[0:n_grid_pts, n_grid_pts:]
inv_cov_data_data, _ = sym_defpos_matrix_inversion_cholesky(cov_data_data)
print("inv-cov-mat is constructed")

beta = np.transpose(np.array(part2_sample_mean[0:6]))
krigging_on_grid = krigging_design_mat@beta 
# print(krigging_on_grid.shape)
print(cov_grid_data.shape, inv_cov_data_data.shape, (np.transpose(np.array(data_soil_carbon))-design_mat_degree1_D1l@beta).shape)
krigging_on_grid += (cov_grid_data @ inv_cov_data_data @ (np.transpose(np.array(data_soil_carbon))-design_mat_degree1_D1l@beta))
print("computing is completed")

krigging_sd_on_grid = cov_grid_grid - cov_grid_data@inv_cov_data_data@np.transpose(cov_grid_data)

axs_eda_1 = axs_eda[1].scatter(*zip(*grid_pts), c=krigging_on_grid, s=50, alpha=0.6, vmin=0, vmax=55, cmap=plt_cm)
axs_eda1_handles, axs_eda1_labels = axs_eda_1.legend_elements(prop="colors", alpha=0.6)
axs_eda[1].legend(axs_eda1_handles, axs_eda1_labels)
axs_eda[1].set_title("predicted")

# axs_eda[2].scatter(*zip(*grid_pts), c=np.diag(krigging_sd_on_grid), s=np.diag(krigging_sd_on_grid))
plt.show()