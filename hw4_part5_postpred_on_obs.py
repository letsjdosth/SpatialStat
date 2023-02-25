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
plt_cm = plt.cm.get_cmap('RdYlBu')
fig_post_pred, axs_post_pred = plt.subplots(1, 3, figsize=(15, 5))
fig_post_pred.tight_layout()
axs_post_pred0 = axs_post_pred[0].scatter(data_long_x, data_lat_y, c=data_soil_carbon, vmin=0, vmax=55)
axs_post_pred[0].set_title("soil carbon")
for i, txt in enumerate(data_landuse_str):
    axs_post_pred[0].annotate(txt, (data_long_x[i], data_lat_y[i]))
axs_post_pred0_handles, axs_post_pred0_labels = axs_post_pred0.legend_elements(prop="colors", alpha=0.6)
axs_post_pred[0].legend(axs_post_pred0_handles, axs_post_pred0_labels)



# ============================================================
# post_predictive dist on observed point
# #                              0        1        2        3        4        5        6           7        8      9    10          11
# part2_inst.set_variable_names(["beta0", "beta1", "beta2", "beta3", "beta4", "beta5", "sigma2_T", "theta", "phi", "v", "sigma2_S", "tau2"])
n_data_pts = len(data_pts)
np_random_inst = np.random.default_rng(20230224)

post_pred_samples_on_obs_pt = []
for i, post_sample in enumerate(part2_inst.MC_sample):
    if i%20==0 and i>0:
        print("iteration", i, "/ 4000")
    # if i==20:
    #     break
    
    matern_inst = Matern(post_sample[9], post_sample[10],post_sample[8])
    matern_cov_mat = matern_inst.cov_matrix(data_pts) #at data points

    post_cov_mat = matern_cov_mat + np.diag([post_sample[11] for _ in range(n_data_pts)])
    # inv_cov_data_data, _ = sym_defpos_matrix_inversion_cholesky(cov_data_data)

    beta = np.transpose(np.array(post_sample[0:6]))
    post_mean = design_mat_degree1_D1l@beta
    
    pred_sample = np_random_inst.multivariate_normal(post_mean, post_cov_mat)
    post_pred_samples_on_obs_pt.append(pred_sample)

post_pred_inst = MCMC_Diag() #abused, but...
post_pred_inst.set_mc_samples_from_list(post_pred_samples_on_obs_pt)
post_pred_inst.write_samples('hw4_fullbayes_pred_samples')

# =====================

post_pred_mean = post_pred_inst.get_sample_mean()
post_pred_var = post_pred_inst.get_sample_var()
post_pred_quantile95 = post_pred_inst.get_sample_quantile([0.025, 0.975])
post_pred_covered = []
for i, (soil, quant) in enumerate(zip(data_soil_carbon, post_pred_quantile95)):
    covered = 'Y' if quant[0] <= soil <= quant[1] else 'N'
    # print(soil, quant, covered)
    post_pred_covered.append(covered)

axs_post_pred1 = axs_post_pred[1].scatter(*zip(*data_pts), c=post_pred_mean, s=50, alpha=0.6, vmin=0, vmax=55)
axs_post_pred1_handles, axs_post_pred1_labels = axs_post_pred1.legend_elements(prop="colors", alpha=0.6)
axs_post_pred[1].legend(axs_post_pred1_handles, axs_post_pred1_labels)
axs_post_pred[1].set_title("predicted")

axs_post_pred2 = axs_post_pred[2].scatter(*zip(*data_pts), c=post_pred_var, s=50, alpha=0.6)
axs_post_pred2_handles, axs_post_pred2_labels = axs_post_pred2.legend_elements(prop="colors", alpha=0.6)
axs_post_pred[2].legend(axs_post_pred2_handles, axs_post_pred2_labels)
axs_post_pred[2].set_title("prediction var")
for i, txt in enumerate(post_pred_covered):
    axs_post_pred[2].annotate(str(txt), (data_long_x[i], data_lat_y[i]))

plt.show()