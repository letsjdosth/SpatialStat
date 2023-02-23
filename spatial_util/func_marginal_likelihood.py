import numpy as np
import scipy

from cov_functions import Matern
from least_squares import OLS_by_QR

def marginal_likelihood(range_phi, smoothness_nu, trend_design_X, resp_Y, data_long, data_lat, nugget=0):
    range_phi = range_phi[0]
    n_data = len(resp_Y)
    matern_scale1_inst = Matern(smoothness_nu, 1, range_phi)
    data_points = [[lon, lat] for lon, lat in zip(data_long, data_lat)]
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

