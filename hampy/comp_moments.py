import numpy as np
from scipy.integrate import simpson as simps

mass_p = 0.010438870      #proton mass in units eV/c^2 where c = 299792 km/s
charge_p = 1              #proton charge in units eV

class hammer_moments:
    def __init__(self, phi_grid, energy_grid, theta_grid):
        self.phi_grid = phi_grid * np.pi / 180.
        self.vel_grid = np.sqrt(2 * charge_p * energy_grid / mass_p)
        self.theta_grid = theta_grid * np.pi / 180.

        self.vx = self.vel_grid * np.cos(self.theta_grid) * np.cos(self.phi_grid)
        self.vy = self.vel_grid * np.cos(self.theta_grid) * np.sin(self.phi_grid)
        self.vz = self.vel_grid * np.sin(self.theta_grid)

        # building the six independent components of the pressure tensor
        self.vxx = self.vx * self.vx
        self.vxy = self.vx * self.vy
        self.vxz = self.vx * self.vz
        self.vyx = self.vy * self.vx
        self.vyy = self.vy * self.vy
        self.vyz = self.vz * self.vz
        self.vzz = self.vz * self.vz
    
    def get_vdf_moments(self, mask, vdf_3d):
        # creating a moments dictionary
        moments = {}

        vdf_3d[:,~mask] = 0.0

        # log_vdf_2d = np.nan_to_num(log_vdf_2d, nan=0.0, posinf=0.0, neginf=0.0)
        # vdf_2d = np.power(10, log_vdf_2d)

        integrand = vdf_3d * np.cos(self.theta_grid) * self.vel_grid**2

        moments['n'] = self.compute_density_moments(integrand)
        moments['Ux'], moments['Uy'], moments['Uz'] = self.compute_velocity_moments(integrand, moments['n'])
        moments['Txx'], moments['Txy'], moments['Txz'],\
        moments['Tyy'], moments['Tyz'], moments['Tzz'] = self.compute_pressure_moments(integrand, moments['n'])

        # fixing the /(2e-5) which should have been /(2e+5) in the vdf calculation in load_data or get_span_data
        moments['n'] = moments['n'] * 1e-10

        return moments

    def compute_density_moments(self, integrand):
        return -simps(simps(simps(integrand, x=self.vel_grid[0,:,0], axis=1),
                            x=self.theta_grid[0,0,:], axis=1), x=self.phi_grid[:,0,0])

    def compute_velocity_moments(self, integrand, density):
        vx = simps(simps(simps(integrand * self.vx, x=self.vel_grid[0,:,0], axis=1),
                           x=self.theta_grid[0,0,:], axis=1), x=self.phi_grid[:,0,0])/density
        vy = simps(simps(simps(integrand * self.vy, x=self.vel_grid[0,:,0], axis=1),
                           x=self.theta_grid[0,0,:], axis=1), x=self.phi_grid[:,0,0])/density
        vz = simps(simps(simps(integrand * self.vz, x=self.vel_grid[0,:,0], axis=1),
                           x=self.theta_grid[0,0,:], axis=1), x=self.phi_grid[:,0,0])/density

        return vx, vy, vz

    def compute_pressure_moments(self, integrand, density):
        Txx = simps(simps(simps(integrand * self.vxx, x=self.vel_grid[0,:,0], axis=1),
                          x=self.theta_grid[0,0,:], axis=1), x=self.phi_grid[:,0,0])/density
        Txy = simps(simps(simps(integrand * self.vxy, x=self.vel_grid[0,:,0], axis=1),
                          x=self.theta_grid[0,0,:], axis=1), x=self.phi_grid[:,0,0])/density
        Txz = simps(simps(simps(integrand * self.vxz, x=self.vel_grid[0,:,0], axis=1),
                          x=self.theta_grid[0,0,:], axis=1), x=self.phi_grid[:,0,0])/density
        Tyy = simps(simps(simps(integrand * self.vyy, x=self.vel_grid[0,:,0], axis=1),
                          x=self.theta_grid[0,0,:], axis=1), x=self.phi_grid[:,0,0])/density
        Tyz = simps(simps(simps(integrand * self.vyz, x=self.vel_grid[0,:,0], axis=1),
                          x=self.theta_grid[0,0,:], axis=1), x=self.phi_grid[:,0,0])/density
        Tzz = simps(simps(simps(integrand * self.vzz, x=self.vel_grid[0,:,0], axis=1),
                          x=self.theta_grid[0,0,:], axis=1), x=self.phi_grid[:,0,0])/density

        return Txx, Txy, Txz, Tyy, Tyz, Tzz
