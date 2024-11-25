import glob
import astropy.units as u
from astropy.coordinates import SkyCoord
from datetime import datetime, timedelta
import sunpy.coordinates
import numpy as np
from sunpy.coordinates import spice, HeliographicStonyhurst, HeliographicCarrington
import matplotlib.pyplot as plt
# import helpers as h
kernel_files = glob.glob("./spice_data/*.bsp")
spice.initialize(kernel_files)
spice.install_frame('IAU_SUN')
plt.style.use("dark_background")

@u.quantity_input
def delta_long(r:u.R_sun,
               r_inner=2.5*u.R_sun,
               vsw=360.*u.km/u.s,
               omega_sun=14.713*u.deg/u.d,
               ):
    """ 
    Ballistic longitudinal shift of a Parker spiral connecting two
    points at radius r and r_inner, for a solar wind speed vsw. Solar
    rotation rate is also tunable
    """
    return (omega_sun * (r - r_inner) / vsw).to("deg")

def ballistically_project(skycoord,r_inner = 2.5*u.R_sun, vr_arr=None) :
    """
    Given a `SkyCoord` of a spacecraft trajectory in the Carrington frame,
    with `representation_type="spherical"`, and optionally an array of
    measured solar wind speeds at the same time intervals of the trajectory,
    return a SkyCoord for the trajectory ballistically projected down to 
    `r_inner` via a Parker spiral of the appropriate curvature. When `vr_arr`
    is not supplied, assumes wind speed is everywhere 360 km/s
    """
    if skycoord.representation_type != "spherical" :
        skycoord.representation_type="spherical"
    if vr_arr is None : vr_arr = np.ones(len(skycoord))*360*u.km/u.s
    lons_shifted = skycoord.lon + delta_long(skycoord.radius,
                                             r_inner=r_inner,
                                             vsw=vr_arr
                                            )
    return SkyCoord(
        lon = lons_shifted, 
        lat = skycoord.lat,
        radius = r_inner * np.ones(len(skycoord)),
        representation_type="spherical",
        frame = skycoord.frame
    )

def get_trajectory(tstart=None, tend=None, dt_mission=None):
    if(tstart is not None):
        time_diff = tend - tstart
        dt_mission = np.arange(tstart, tend + timedelta(days=1), time_diff/100).astype(datetime)
    else:
        dt_mission = dt_mission


    # Generate Positions for Parker Solar Probe
    parker_trajectory_inertial = spice.get_body('SPP', dt_mission)
    parker_trajectory_carrington = parker_trajectory_inertial.transform_to(HeliographicCarrington(observer="self"))
    parker_trajectory_inertial.representation_type = "spherical"
    parker_trajectory_carrington.representation_type = "spherical"

    return parker_trajectory_carrington

    '''
    # Generate Trajectory for Solar Orbiter
    solo_trajectory_inertial = spice.get_body('SOLO', dt_mission)
    solo_trajectory_carrington = solo_trajectory_inertial.transform_to(HeliographicCarrington(observer="self"))
    solo_trajectory_inertial.representation_type = "cartesian"
    solo_trajectory_carrington.representation_type = "cartesian"

    # Generate Trajectory for Earth
    earth_trajectory_inertial = spice.get_body('Earth', dt_mission)
    earth_trajectory_carrington = earth_trajectory_inertial.transform_to(HeliographicCarrington(observer="self"))
    earth_trajectory_inertial.representation_type = "cartesian"
    earth_trajectory_carrington.representation_type = "cartesian"

    parker_projected = ballistically_project(parker_trajectory_carrington)
    solo_projected = ballistically_project(solo_trajectory_carrington)
    earth_projected = ballistically_project(earth_trajectory_carrington)
    '''