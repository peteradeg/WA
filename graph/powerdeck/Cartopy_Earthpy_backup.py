# Working in jupyter notebook
# does it work in ipython

# easier earhtpy
#http://earthpy.org/tag/cartopy.html

# at this point we can change projections etc. 

from netCDF4 import Dataset, MFDataset, num2date
import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm
import cartopy.crs as ccrs

import sys
import os
from cartopy.util import add_cyclic_point

## Temperature
# Load data
#http://earthpy.org/cartopy_backgroung.html


#for year in range()
flf = Dataset('/mnt/y/Data/Weather/ECMWF/Data_R/ml/2010/01/era5_hourly_ml_AUS_201001.nc')
lat = flf.variables['latitude'][:]
lon = flf.variables['longitude'][:]


temp = flf.variables['t2m'][0,:,:]
# need to convert this into datetime?


plt.figure(figsize=(13,6.2))
    
ax = plt.subplot(111, projection=ccrs.PlateCarree())

mm = ax.pcolormesh(lon,\
				   lat,\
                   temp,\
                   vmin=273,\
                   vmax=310,\
                   transform=ccrs.PlateCarree(),cmap="RdBu_r" )

ax.coastlines();
plt.savefig("matplotlib.png")