from math import exp, sin, gamma
from random import seed

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import kv

class IsotropicCovBase:
    def __init__(self, scale_sigma2: float, range_phi: float):
        self.sigma2 = scale_sigma2
        self.phi = range_phi
        self.pseudo_rv_generator = np.random.default_rng()

    def _check_nonnegative(self, dist):
        if dist<0:
            raise ValueError("dist should be nonnegative.")

    def covariance_function(self, dist):
        pass
    
    def semi_variogram(self, dist):
        pass

    def plot_covariance(self, start, end, by, show=True):
        grid = np.arange(start, end, by)
        cov_on_grid = []
        for x in grid:
            cov_on_grid.append(self.covariance_function(x))
        plt.plot(grid, cov_on_grid)
        if show:
            plt.show()

    def plot_semi_variogram(self, start, end, by, show=True):
        grid = np.arange(start, end, by)
        semi_var_on_grid = []
        for x in grid:
            semi_var_on_grid.append(self.semi_variogram(x))
        plt.plot(grid, semi_var_on_grid)
        if show:
            plt.show()

    def _dist_euclidean(self, loc1:list[float], loc2:list[float]):
        if len(loc1)!=len(loc2):
            raise ValueError("the dimensions of two location indices should be the same")
        dist2 = 0
        for s1, s2 in zip(loc1, loc2):
            dist2 += (s1-s2)**2
        return dist2**0.5

    def cov_matrix(self, points, topology="Euclidean"):
        cov_mat = []
        for i in points:
            cov_mat.append([])
            for j in points:
                dist = 0
                if topology == "Euclidean":
                    dist = self._dist_euclidean(i, j)
                else:
                    raise ValueError("now, only Euclidean topology can be used.")
                cov_mat[-1].append(self.covariance_function(dist))
        return np.array(cov_mat)

    def sampler(self, points, topology="Euclidean"):
        cov_mat = self.cov_matrix(points, topology)
        sample = self.pseudo_rv_generator.multivariate_normal(np.zeros(len(points)), cov_mat)
        return sample


class Spherical(IsotropicCovBase):
    def covariance_function(self, dist):
        self._check_nonnegative(dist)
        cov = 0
        if dist == 0:
            cov = self.sigma2
        elif dist < self.phi:
            cov = self.sigma2 * (1 - 3*dist / (2*self.phi) + dist**3/(2*self.phi**3))
        else:
            cov = 0
        return cov

    def semi_variogram(self, dist):
        self._check_nonnegative(dist)
        semi_var = 0
        if dist <= self.phi:
            semi_var = self.sigma2 * (3*dist/(2*self.phi) - dist**3/(2*self.phi**3))
        else:
            semi_var = self.sigma2
        return semi_var
   
class PoweredExp(IsotropicCovBase):
    def __init__(self, order_nu: float, scale_sigma2: float, range_phi: float):
        if 0 < order_nu <= 2:
            self.nu = order_nu
        else:
            raise ValueError("It should be 0<nu<=2.")
        super().__init__(scale_sigma2, range_phi)

    def covariance_function(self, dist):
        self._check_nonnegative(dist)
        cov = self.sigma2 * exp(-(abs(dist/self.phi)**self.nu))
        return cov

    def semi_variogram(self, dist):
        self._check_nonnegative(dist)
        semi_var = self.sigma2 * (1 - exp(-(abs(dist/self.phi)**self.nu)))
        return semi_var

class RationalQuadratic(IsotropicCovBase):
    def covariance_function(self, dist):
        self._check_nonnegative(dist)
        cov = self.sigma2 * (1 - dist**2/(self.phi**2 + dist**2))
        return cov

    def semi_variogram(self, dist):
        self._check_nonnegative(dist)
        semi_var = self.sigma2*(dist**2) / (self.phi**2 + dist**2)
        return semi_var

class Wave(IsotropicCovBase):
    def covariance_function(self, dist):
        self._check_nonnegative(dist)
        if dist==0:
            cov = self.sigma2
        else:
            cov = self.sigma2 * sin(dist/self.phi) / (dist/self.phi)
        return cov

    def semi_variogram(self, dist):
        self._check_nonnegative(dist)
        if dist==0:
            semi_var = 0
        else:
            semi_var = self.sigma2 * (1 - sin(dist/self.phi) / (dist/self.phi))
        return semi_var

class Matern(IsotropicCovBase):
    def __init__(self, order_nu: float, scale_sigma2: float, range_phi: float):
        if 0 < order_nu:
            self.nu = order_nu
        else:
            raise ValueError("It should be 0 < nu.")
        super().__init__(scale_sigma2, range_phi)

    def covariance_function(self, dist):
        self._check_nonnegative(dist)
        if dist==0:
            cov = self.sigma2
        else:
            cov = (self.sigma2 / (2**(self.nu-1)*gamma(self.nu))) * ((dist/self.phi)**self.nu) * kv(self.nu, dist/self.phi)
        return cov

    def semi_variogram(self, dist):
        self._check_nonnegative(dist)
        if dist==0:
            semi_var = 0
        else:
            semi_var = self.sigma2 *(1 - (1/(2**(self.nu-1)*gamma(self.nu))) * ((dist/self.phi)**self.nu) * kv(self.nu, dist/self.phi))
        return semi_var

if __name__=="__main__":
    # def find_05_phi():
    #     for phi in np.arange(0.0001, 5, 0.0001):
    #         inst = Wave(1, phi)
    #         corr_at_1 = inst.covariance_function(1)/inst.covariance_function(0)
    #         if corr_at_1>0.05:
    #             print(phi, corr_at_1)
    #             return phi
    # find_05_phi()

    seed(20230122)
    sigma2 = 1
    grid_for_sample = [np.array([i/10]) for i in range(100)]

    spherical_phi = 1.2325
    spherical_inst = Spherical(sigma2, spherical_phi)
    print(spherical_inst.covariance_function(1)/spherical_inst.covariance_function(0))
    spherical_inst.plot_covariance(0, 1.5, 0.0001)
    spherical_inst.plot_semi_variogram(0, 1.5, 0.0001)
    spherical_sample = spherical_inst.sampler(grid_for_sample)
    plt.plot(grid_for_sample, spherical_sample)
    plt.show()


    poweredExp_1_phi = 0.3339
    poweredExp_1_inst = PoweredExp(1, sigma2, poweredExp_1_phi)
    print(poweredExp_1_inst.covariance_function(1)/poweredExp_1_inst.covariance_function(0))
    poweredExp_1_inst.plot_covariance(0, 1.5, 0.0001)
    poweredExp_1_inst.plot_semi_variogram(0, 1.5, 0.0001)
    poweredExp_1_sample = poweredExp_1_inst.sampler(grid_for_sample)
    plt.plot(grid_for_sample, poweredExp_1_sample)
    plt.show()



    rationalQuadratic_phi = 0.2295
    rationalQuadratic_inst = RationalQuadratic(sigma2, rationalQuadratic_phi)
    print(rationalQuadratic_inst.covariance_function(1)/rationalQuadratic_inst.covariance_function(0))
    rationalQuadratic_inst.plot_covariance(0, 1.5, 0.0001)
    rationalQuadratic_inst.plot_semi_variogram(0, 1.5, 0.0001)
    rationalQuadratic_sample = rationalQuadratic_inst.sampler(grid_for_sample)
    plt.plot(grid_for_sample, rationalQuadratic_sample)
    plt.show()

    wave_phi = 0.0673
    wave_inst = Wave(sigma2, wave_phi)
    print(wave_inst.covariance_function(1)/wave_inst.covariance_function(0))
    wave_inst.plot_covariance(0, 1.5, 0.0001)
    wave_inst.plot_semi_variogram(0, 1.5, 0.0001)
    wave_sample = wave_inst.sampler(grid_for_sample)
    plt.plot(grid_for_sample, wave_sample)
    plt.show()

    matern_1_phi = 0.2501
    matern_1_inst = Matern(1, sigma2, matern_1_phi)
    print(matern_1_inst.covariance_function(1)/matern_1_inst.covariance_function(0))
    matern_1_inst.plot_covariance(0, 1.5, 0.0001)
    matern_1_inst.plot_semi_variogram(0, 1.5, 0.0001)
    matern_1_sample = matern_1_inst.sampler(grid_for_sample)
    plt.plot(grid_for_sample, matern_1_sample)
    plt.show()