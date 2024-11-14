import numpy as np
from scipy.integrate import simpson as simps

class hammer_moments:
    def __init__(self, vel_grid, theta_grid):
        self.vel_grid = vel_grid
        self.theta_grid = theta_grid

        self.vx = vel_grid * np.cos(theta_grid)
        self.vy = vel_grid * np.cos(theta_grid)
        self.vz = vel_grid * np.sin(theta_grid)

        # building the six independent components of the pressure tensor
        self.vxx = self.vx * self.vx
        self.vxy = self.vx * self.vy
        self.vxz = self.vx * self.vz
        self.vyx = self.vy * self.vx
        self.vyy = self.vy * self.vy
        self.vyz = self.vz * self.vz
        self.vzz = self.vz * self.vz

        # creating a moments dictionary
        self.moments = {}
    
    def get_vdf_momnets(self, log_vdf_2d):
        vdf_2d = np.power(10, log_vdf_2d)

        integrand = vdf_2d * self.vel_grid**2

        self.moments['rho'] = self.compute_density_moments(integrand)
        self.moments['Ux'], self.moments['Uy'], self.moments['Uz'] =\
                             self.compute_velocity_moments(integrand)
        self.moments['Txx'], self.moments['Txy'], self.moments['Txz'],\
        self.moments['Tyy'], self.moments['Tyz'], self.moments['Tzz'] =\
                             self.compute_pressure_moments(integrand)

    def compute_density_moments(self, integrand):
        return simps(simps(integrand, x=self.vel_grid[:,0], axis=0), x=self.theta_grid[0,:])

    def compute_velocity_moments(self, integrand):
        vx = simps(simps(integrand * self.vx, x=self.vel_grid[:,0], axis=0), x=self.theta_grid[0,:])
        vy = simps(simps(integrand * self.vy, x=self.vel_grid[:,0], axis=0), x=self.theta_grid[0,:])
        vz = simps(simps(integrand * self.vz, x=self.vel_grid[:,0], axis=0), x=self.theta_grid[0,:])

        return vx, vy, vz

    def compute_pressure_moments(self, integrand):
        Txx = simps(simps(integrand * self.vxx, x=self.vel_grid[:,0], axis=0), x=self.theta_grid[0,:])
        Txy = simps(simps(integrand * self.vxy, x=self.vel_grid[:,0], axis=0), x=self.theta_grid[0,:])
        Txz = simps(simps(integrand * self.vxz, x=self.vel_grid[:,0], axis=0), x=self.theta_grid[0,:])
        Tyy = simps(simps(integrand * self.vyy, x=self.vel_grid[:,0], axis=0), x=self.theta_grid[0,:])
        Tyz = simps(simps(integrand * self.vyz, x=self.vel_grid[:,0], axis=0), x=self.theta_grid[0,:])
        Tzz = simps(simps(integrand * self.vzz, x=self.vel_grid[:,0], axis=0), x=self.theta_grid[0,:])

        return Txx, Txy, Txz, Tyy, Tyz, Tzz
