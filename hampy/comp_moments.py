import numpy as np
from scipy.integrate import simpson as simps
import astropy.constants as c
import astropy.units as u

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

    def build_vv_for_pressure_moment(self, ux, uy, uz):
        # building the six independent components of the pressure tensor
        self.vxx = (self.vx - ux) * (self.vx - ux)
        self.vxy = (self.vx - ux) * (self.vy - uy)
        self.vxz = (self.vx - ux) * (self.vz - uz)
        self.vyy = (self.vy - uy) * (self.vy - uy)
        self.vyz = (self.vy - uy) * (self.vz - uz)
        self.vzz = (self.vz - uz) * (self.vz - uz)

    
    def get_vdf_moments(self, mask, vdf_3d):
        # creating a moments dictionary
        moments = {}

        vdf_3d[:,~mask] = 0.0

        # log_vdf_2d = np.nan_to_num(log_vdf_2d, nan=0.0, posinf=0.0, neginf=0.0)
        # vdf_2d = np.power(10, log_vdf_2d)

        integrand = vdf_3d * np.cos(self.theta_grid) * self.vel_grid**2

        moments['n'] = self.compute_density_moments(integrand)
        moments['Ux'], moments['Uy'], moments['Uz'] = self.compute_velocity_moments(integrand, moments['n'])

        self.build_vv_for_pressure_moment(moments['Ux'], moments['Uy'], moments['Uz'])

        moments['Txx'], moments['Txy'], moments['Txz'],\
        moments['Tyy'], moments['Tyz'], moments['Tzz'] = self.compute_pressure_moments(integrand, moments['n'])

        # fixing the /(2e-5) which should have been /(2e+5) in the vdf calculation in load_data or get_span_data
        moments['n'] = moments['n'] * 1e-10

        return moments

    def compute_density_moments(self, integrand):
        return simps(simps(simps(integrand, x=self.vel_grid[0,:,0], axis=1),
                            x=self.theta_grid[0,0,:], axis=1)[::-1], x=self.phi_grid[::-1,0,0])

    def compute_velocity_moments(self, integrand, density):
        vx = simps(simps(simps(integrand * self.vx, x=self.vel_grid[0,:,0], axis=1),
                           x=self.theta_grid[0,0,:], axis=1)[::-1], x=self.phi_grid[::-1,0,0])/density
        vy = simps(simps(simps(integrand * self.vy, x=self.vel_grid[0,:,0], axis=1),
                           x=self.theta_grid[0,0,:], axis=1)[::-1], x=self.phi_grid[::-1,0,0])/density
        vz = simps(simps(simps(integrand * self.vz, x=self.vel_grid[0,:,0], axis=1),
                           x=self.theta_grid[0,0,:], axis=1)[::-1], x=self.phi_grid[::-1,0,0])/density

        return vx, vy, vz

    def compute_pressure_moments(self, integrand, density, return_temp_moments=True):
        Pxx = simps(simps(simps(integrand * self.vxx, x=self.vel_grid[0,:,0], axis=1),
                          x=self.theta_grid[0,0,:], axis=1)[::-1], x=self.phi_grid[::-1,0,0])
        Pxy = simps(simps(simps(integrand * self.vxy, x=self.vel_grid[0,:,0], axis=1),
                          x=self.theta_grid[0,0,:], axis=1)[::-1], x=self.phi_grid[::-1,0,0])
        Pxz = simps(simps(simps(integrand * self.vxz, x=self.vel_grid[0,:,0], axis=1),
                          x=self.theta_grid[0,0,:], axis=1)[::-1], x=self.phi_grid[::-1,0,0])
        Pyy = simps(simps(simps(integrand * self.vyy, x=self.vel_grid[0,:,0], axis=1),
                          x=self.theta_grid[0,0,:], axis=1)[::-1], x=self.phi_grid[::-1,0,0])
        Pyz = simps(simps(simps(integrand * self.vyz, x=self.vel_grid[0,:,0], axis=1),
                          x=self.theta_grid[0,0,:], axis=1)[::-1], x=self.phi_grid[::-1,0,0])
        Pzz = simps(simps(simps(integrand * self.vzz, x=self.vel_grid[0,:,0], axis=1),
                          x=self.theta_grid[0,0,:], axis=1)[::-1], x=self.phi_grid[::-1,0,0])

        conv_factor = 1.0

        if(return_temp_moments):
            # calculating the conversion factor to get temperature in units of 1e6 K
            conv_factor = c.m_p.value / (density * c.k_B.value)

        return Pxx * conv_factor, Pxy * conv_factor, Pxz * conv_factor, Pyy * conv_factor, Pyz * conv_factor, Pzz * conv_factor
