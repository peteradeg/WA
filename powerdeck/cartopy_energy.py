# Plotting suggestions https://scitools.org.uk/iris/docs/v2.0/gallery.html 
# non standard projections https://scitools.org.uk/iris/docs/v2.0/examples/General/projections_and_annotations.html#projections-and-annotations-01

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

flf = Dataset('/home/peter/weatheranalytics/Data/ECMWF/2017/01/era5_hourly_surface_AUS_201701.nc')
lat = flf.variables['latitude'][:]
lon = flf.variables['longitude'][:]


temp = flf.variables['t2m'][0,:,:]



plt.figure(figsize=(13,6.2))
    
ax = plt.subplot(111, projection=ccrs.PlateCarree())

mm = ax.pcolormesh(lon,\
				   lat,\
                   temp,\
                   vmin=273,\
                   vmax=310,\
                   transform=ccrs.PlateCarree(),cmap="RdBu_r" )

ax.coastlines();



## Wind
#https://scitools.org.uk/iris/docs/v2.0/examples/Meteorology/wind_speed.html#wind-speed-00

import matplotlib.pyplot as plt
import numpy as np

import iris
import iris.coord_categorisation
import iris.quickplot as qplt

import cartopy
import cartopy.feature as cfeat
import cartopy.crs as ccrs



# Load the u and v components of wind from a pp file
#infile = iris.sample_data_path('home/peter/weatheranalytics/Data/ECMWF/2017/01/era5_hourly_surface_AUS_201701.nc')
infile = '/home/peter/weatheranalytics/Data/ECMWF/2017/01/era5_hourly_surface_AUS_201701.nc'

wind = iris.load(infile)

uwind = wind[0][0,:,:]
vwind = wind[2][0,:,:]

ulon = uwind.coord('longitude')
vlon = vwind.coord('longitude')

# Create a cube containing the wind speed
windspeed = (uwind ** 2 + vwind ** 2) ** 0.5
windspeed.rename('windspeed')

x = ulon.points
y = uwind.coord('latitude').points
u = uwind.data
v = vwind.data

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
qplt.contourf(windspeed, 20)
plt.quiver(x, y, u*100, v*100, pivot='middle')
# Plot the wind speed as a contour plot
#qplt.contourf(windspeed, 20)

# # Normalise the data for uniform arrow size
# u_norm = u / np.sqrt(u ** 2.0 + v ** 2.0)
# v_norm = v / np.sqrt(u ** 2.0 + v ** 2.0)

# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())


# qplt.contourf(windspeed, 20)
# # current_map = iplt.gcm()
# # current_map.drawcoastlines()
# plt.quiver(x, y, u_norm, v_norm, pivot='middle')#, transform=transform)

plt.title("Wind speed over Lake Victoria")
qplt.show()