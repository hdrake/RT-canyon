from xmitgcm import open_mdsdataset
from xgcm import Grid
import numpy as np

def load_RT_canyon_hydrostatic():
    data_dir = "/pool001/xruan/RT_canyon_hydrostatic/diag/"
    grid_dir = "/pool001/xruan/RT_canyon_hydrostatic/input/"
    ds = open_mdsdataset(
        data_dir, grid_dir = grid_dir, iters="all", prefix=["TS_inst", "tracer_inst"],
        ignore_unknown_vars=True, delta_t = 5.
    ).drop(["maskC", "maskInC"]).drop_dims(["Zl", "Zu", "Zp1", "YG", "XG"])
    grid = Grid(ds, periodic=['X', 'Y'])

    # Convert elapsed time to actual datetime and include release time for reference
    ds['time'] = ds['time'] + np.datetime64('2021-06-27T00:00:00') # Beginning of simulation
    ds['THETA'].attrs['units'] = '°C'
    ds.attrs['release_time'] = '2021-07-01T04:00:00' # Should actually be T03:00:00 according to cruise report
    ds.attrs['dye_release_loc'] = (-11.9097, 54.2221) # From Cruise Report
    ds = ds.assign_coords({'hours_since_release': ((ds['time'] - np.datetime64(ds.release_time))/(1.e9)).astype('float64')/3600.})
    
    # Total mass of injected dye
    Vdyemixture = 149. * 1.e-3 # [L dye mix] * (m^3/L) = [m^3 dye mix]
    Cdye = 40./100. # [m^3 dye]/[m^3 dye mix]
    Vdye = Cdye*Vdyemixture # [m^3 dye]/[m^3 dye mix] * [m^3 dye mix] = [m^3 dye]

    ρdye = 1.602 * 1.e3 * 1.e3 # [g/mL dye] * (mL/L) * (L/m^3) = [g/m^3 dye]
    Mdye = Vdye*ρdye # [m^3 dye] * [g/m^3 dye] = [g dye]
    
    # Total mass of simulated dye
    ρ0 = 1030. * 1.e3 # [kg/m^3 SW] * g/kg = [g/m^3 SW]
    ds['dm'] = (ds.rA*ds.drF*ds.hFacC) * ρ0 # [m^2*m*1 SW] * [k/m^3 SW] = [g SW]
    Msim = (ds.TRAC01*ds.dm).sum(['XC', 'YC', 'Z']).isel(time=0).values # ([g sim dye]/[g SW]) * [g SW] = [g sim dye]
    
    ds['TRAC01'] = (ds.TRAC01/Msim) * Mdye * 1.e9 #  ([g sim dye]/[g SW]) / [g sim dye] * [g dye] * 1e9 = [g sim dye]/[g SW] * 1e9 = ppb dye
    
    return ds, grid