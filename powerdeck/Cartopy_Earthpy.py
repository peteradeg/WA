# easier earhtpy
#http://earthpy.org/tag/cartopy.html

from netCDF4 import Dataset, MFDataset, num2date
import matplotlib.pylab as plt
#%matplotlib inline
import numpy as np
from matplotlib import cm
import cartopy.crs as ccrs
#from cmocean import cm as cmo

import sys
import os
from cartopy.util import add_cyclic_point

#flf = Dataset('./temperature_annual_1deg.nc')

flf = Dataset('/mnt/y/Data/Weather/ECMWF/Data_R/ml/2010/01/era5_hourly_ml_AUS_201001.nc')

lat = flf.variables['latitude'][:]
lon = flf.variables['longitude'][:]

temp = flf.variables['t2m'][0,:,:]

plt.figure(figsize=(13,6.2))
    
#ax = plt.subplot(111, projection=ccrs.PlateCarree())
ax = plt.subplot(111)#, projection=ccrs.PlateCarree())

mm = ax.pcolormesh(lon,\
                   lat,\
                   temp,\
                   vmin=-2,\
                   vmax=30)#transform=ccrs.PlateCarree(),cmap=cmo.thermal )
ax.coastlines();