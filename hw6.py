import numpy as np
import matplotlib.pyplot as plt

def bazier_kernel(s_diff, v_smoothness, ker_range=1):
    norm_sq_s = np.sum([s**2 for s in s_diff])
    scaled_norm_sq_s = norm_sq_s/(ker_range**2)
    if scaled_norm_sq_s < 1:
        return (1 - scaled_norm_sq_s)**v_smoothness
    else:
        return 0

def generate_kernel_matrix(data_long_lat:list[tuple], grid_long_lat:list[tuple], v_smoothness, ker_range=1):
    ker_mat = []
    for data_pt in data_long_lat:
        ker_mat.append([])
        for grid_pt in grid_long_lat:
            s_diff = [s1-s2 for s1, s2 in zip(data_pt, grid_pt)]
            k = bazier_kernel(s_diff, v_smoothness, ker_range)
            ker_mat[-1].append(k)
    return np.array(ker_mat)


class GaussianMarkovRandomField_simulator:
    def __init__(self, seed_val):
        self.W_precision = None
        self.np_rand_inst = np.random.default_rng(seed_val)
    def set_precision_W(self, W_neighborhood_based_precision: np.ndarray):
        self.W_precision = W_neighborhood_based_precision
    def set_rectangular_grid(self, vertical_knot_num: int, horizontal_knot_num: int):
        knot_num = vertical_knot_num*horizontal_knot_num
        self.W_precision = np.zeros((knot_num, knot_num),"float64")
        for i in range(knot_num):
            for j in range(knot_num):
                if (i+1)==j and i//horizontal_knot_num==j//horizontal_knot_num:
                    self.W_precision[i,j] = -1
                elif (i-1)==j and i//horizontal_knot_num==j//horizontal_knot_num:
                    self.W_precision[i,j] = -1
                elif (i+horizontal_knot_num)==j or (i-horizontal_knot_num)==j:
                    self.W_precision[i,j] = -1
        for i in range(knot_num):
            self.W_precision[i,i] = -sum(self.W_precision[i,]) + 0.01 #avoid singularity

    def draw_one_realization(self, lambda_scaler = 1):
        knot_num = self.W_precision.shape[0]
        sample = self.np_rand_inst.multivariate_normal(np.zeros(knot_num), np.linalg.inv(lambda_scaler*(self.W_precision)))
        # is it the best way?
        return sample
    
def get_quick_rectangular_grid(vertical_knot_num, horizontal_knot_num, knot_gap:float = 1):
    grid = []
    for i in range(vertical_knot_num):
        for j in range(horizontal_knot_num):
            grid.append((i*knot_gap,j*knot_gap))
    return grid
        
if __name__=="__main__":
    # simulate data from process convolution based on GMRF
    np_random_inst_on_main = np.random.default_rng(20230322)
    gmrf_simulator_inst = GaussianMarkovRandomField_simulator(20230322)
    knots_num_v = 10
    knots_num_h = 10

    gmrf_simulator_inst.set_rectangular_grid(knots_num_v, knots_num_h)
    # print(gmrf_simulator_inst.W_precision)

    lambda_X = 1
    gmrf_sample = gmrf_simulator_inst.draw_one_realization(lambda_scaler=lambda_X)
    gmrf_grid = get_quick_rectangular_grid(knots_num_v, knots_num_h, 1/10)

    y_loc = [(np_random_inst_on_main.random(), np_random_inst_on_main.random()) for _ in range(300)]
    K_ker_mat = generate_kernel_matrix(y_loc, gmrf_grid, v_smoothness = 1, ker_range=0.4)
    lambda_y = 1
    y_val = K_ker_mat @ gmrf_sample + np_random_inst_on_main.multivariate_normal(np.zeros(K_ker_mat.shape[0]), np.eye(K_ker_mat.shape[0])/lambda_y)

    plt.scatter(*zip(*gmrf_grid), c=gmrf_sample, marker=5)
    sct_true = plt.scatter(*zip(*y_loc), c=y_val, s=1.5)
    handles_true, labels_true = sct_true.legend_elements(prop="colors", alpha=0.6)
    plt.legend(handles_true, labels_true, loc=2)
    plt.show()

    # ===

    # fit reversely
    from pyBayes.MCMC_Core import MCMC_Gibbs, MCMC_Diag
    class fitter(MCMC_Gibbs):
        def __init__(self, initial, y_obs, K_kernel_mat, W_nbhd_prec_mat, hyper_aX, hyper_bX, hyper_aY, hyper_bY, seed_val = 20230322):
            self.MC_sample = [initial]
            self.y_obs = y_obs
            self.K_ker_mat = K_kernel_mat
            self.KtK = np.transpose(self.K_ker_mat)@self.K_ker_mat
            self.inv_KtK = np.linalg.inv(self.KtK)
            self.W_prec = W_nbhd_prec_mat
            self.hyper_aX = hyper_aX
            self.hyper_bX = hyper_bX
            self.hyper_aY = hyper_aY
            self.hyper_bY = hyper_bY

            self.n = K_ker_mat.shape[0]
            self.p = K_ker_mat.shape[1]

            self.np_random_inst = np.random.default_rng(seed_val)


        def sampler(self, **kwargs):
            last = self.MC_sample[-1]
            new = self.deep_copier(last)
            #update new
            
            new = self.full_conditional_sampler_X(new)
            new = self.full_conditional_sampler_lambdaX(new)
            new = self.full_conditional_sampler_lambdaY(new)
            self.MC_sample.append(new)
        
        def full_conditional_sampler_X(self, last_param):
            #param
            # 0  1         2
            #[X, lambda_X, lambda_Y]
            new_sample = [x for x in last_param]
            #update new

            prec_mat = last_param[2]*self.KtK + last_param[1]*self.W_prec
            cov_mat = np.linalg.inv(prec_mat)
            mean_vec = cov_mat @ (last_param[2] * np.transpose(self.K_ker_mat) @ self.y_obs)
            new_X = self.np_random_inst.multivariate_normal(mean_vec, cov_mat)
            new_sample[0] = new_X
            return new_sample

        def full_conditional_sampler_lambdaX(self, last_param):
            #param
            # 0  1         2
            #[X, lambda_X, lambda_Y]
            new_sample = [x for x in last_param]
            shape = self.hyper_aX + self.p/2
            rate = self.hyper_bX + 0.5*np.transpose(last_param[0]) @ self.W_prec @ last_param[0]
            new_lambda_X = self.np_random_inst.gamma(shape, 1/rate)
            new_sample[1] = new_lambda_X
            return new_sample
        
        def full_conditional_sampler_lambdaY(self, last_param):
            #param
            # 0  1         2
            #[X, lambda_X, lambda_Y]
            new_sample = [x for x in last_param]
            shape = self.hyper_aY + self.n/2
            err = self.y_obs - self.K_ker_mat @ last_param[0]
            rate = self.hyper_bY + 0.5*np.transpose(err) @ err
            new_lambda_Y = self.np_random_inst.gamma(shape, 1/rate)
            new_sample[2] = new_lambda_Y
            return new_sample
    
    true_setting = True
    if true_setting:
        fit_knots_num_v = 10
        fit_knots_num_h = 10
        fit_grid = get_quick_rectangular_grid(fit_knots_num_v, fit_knots_num_h, 1/fit_knots_num_h)
        fit_K_ker_mat = generate_kernel_matrix(y_loc, fit_grid, 1, 0.4)
        fit_W = gmrf_simulator_inst.W_precision
    else:
        fit_knots_num_v = 15
        fit_knots_num_h = 15
        fit_grid = get_quick_rectangular_grid(fit_knots_num_v, fit_knots_num_h, 1/fit_knots_num_h)
        fit_K_ker_mat = generate_kernel_matrix(y_loc, fit_grid, 1, 0.4)
        fit_simul_inst = GaussianMarkovRandomField_simulator(20230323)
        fit_simul_inst.set_rectangular_grid(fit_knots_num_v, fit_knots_num_h)
        fit_W = fit_simul_inst.W_precision
    

    gibbs_inst = fitter([np.zeros(fit_knots_num_v*fit_knots_num_h), 1, 1], y_val, fit_K_ker_mat, fit_W,
                        0.01, 0.01, 0.01, 0.01, 20230322)
    gibbs_inst.generate_samples(1000)
    X_samples = [x[0] for x in gibbs_inst.MC_sample]
    lambda_samples = [[x[1], x[2]] for x in gibbs_inst.MC_sample]
    diag_inst2 = MCMC_Diag()
    diag_inst2.set_mc_samples_from_list(lambda_samples)
    diag_inst2.set_variable_names(["lambda_X", "lambda_Y"])
    diag_inst2.burnin(300)
    diag_inst2.show_traceplot((1,2))
    diag_inst2.show_hist((1,2))
    diag_inst2.show_acf(30,(1,2))
    diag_inst2.print_summaries(6)

    diag_inst1 = MCMC_Diag()
    diag_inst1.set_mc_samples_from_list(X_samples)
    diag_inst1.set_variable_names(["x"+str(i) for i in range(1, knots_num_v*knots_num_h+1)])
    diag_inst1.burnin(300)
    # diag_inst1.show_traceplot((1,3), [0,1,2])
    # diag_inst1.show_hist((1,3), [0,1,2])
    # diag_inst1.show_acf(30, (1,3), [0,1,2])
    
    posterior_x_mean = diag_inst1.get_sample_mean()
    posterior_x_var = diag_inst1.get_sample_var()

    # ===

    s_new_pts = get_quick_rectangular_grid(31, 31, 1/31)
    K_for_krigging = generate_kernel_matrix(s_new_pts, fit_grid, v_smoothness = 1, ker_range=0.4)
    pred_sample_vec = []
    for x, lamb_vec in zip(diag_inst1.MC_sample, diag_inst2.MC_sample):
        pred_sample = K_for_krigging @ x + np_random_inst_on_main.multivariate_normal(np.zeros(len(s_new_pts)), np.eye(len(s_new_pts))/lamb_vec[1])
        pred_sample_vec.append(pred_sample)
    
    pred_inst1 = MCMC_Diag()
    pred_inst1.set_mc_samples_from_list(pred_sample_vec)
    pred_y_mean = pred_inst1.get_sample_mean()
    pred_y_var = pred_inst1.get_sample_var()
        
    plt.scatter(*zip(*fit_grid), c=posterior_x_mean, marker=5)
    sct_m = plt.scatter(*zip(*s_new_pts), c=pred_y_mean, s=1.5)
    handles_m, labels_m = sct_m.legend_elements(prop="colors", alpha=0.6)
    plt.legend(handles_m, labels_m)
    plt.title("predictive mean")
    plt.show()

    # plt.scatter(*zip(*fit_grid), c=posterior_x_var, marker=5)
    sct_v = plt.scatter(*zip(*s_new_pts), c=pred_y_var, s=1.5)
    handles_v, labels_v = sct_v.legend_elements(prop="colors", alpha=0.6)
    plt.legend(handles_v, labels_v)
    plt.title("predictive variance")
    plt.show()
    
    # ===

    s_original_pts = y_loc
    K_for_krigging2 = generate_kernel_matrix(y_loc, fit_grid, v_smoothness = 1, ker_range=0.4)
    pred_sample_vec2 = []
    for x, lamb_vec in zip(diag_inst1.MC_sample, diag_inst2.MC_sample):
        pred_sample2 = K_for_krigging2 @ x + np_random_inst_on_main.multivariate_normal(np.zeros(len(s_original_pts)), np.eye(len(s_original_pts))/lamb_vec[1])
        pred_sample_vec2.append(pred_sample2)
    
    pred_inst2 = MCMC_Diag()
    pred_inst2.set_mc_samples_from_list(pred_sample_vec2)
    pred_y_mean2 = pred_inst2.get_sample_mean()
    pred_y_var2 = pred_inst2.get_sample_var()
        
    sct_e = plt.scatter(*zip(*s_original_pts), c=(np.array(y_val)-np.array(pred_y_mean2))**2, s=1.5)
    handles_e, labels_e = sct_e.legend_elements(prop="colors", alpha=0.6)
    plt.legend(handles_e, labels_e)
    plt.title("squared error")
    plt.show()

