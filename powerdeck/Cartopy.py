# Plotting with Cartopy
# https://unidata.github.io/python-gallery/examples/xarray_500hPa_map.html

from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import xarray as xr


## Filepath
year=2012
month=1
# Need loop here
data = xr.open_dataset('/mnt/y/Data/Weather/ECMWF/Data_R/ml/{0}/{1:02d}/era5_hourly_ml_AUS_{0}{1:02d}.nc'.format(year,month))

#data = xr.open_mfdataset('/mnt/y/Data/Weather/ECMWF/Data_R/ml/*/*/era5_hourly_ml_AUS_*.nc'.format(year,month))

# X, Y values are in units of km, need them in meters for plotting/calculations
data.x.values = data.x.values * 1000.
data.y.values = data.y.values * 1000.

x, y = np.meshgrid(data.x.values, data.y.values)

## Selecting Data

hght_500 = data.Geopotential_height_isobaric.sel(time1=vtimes[0], isobaric=500)
uwnd_500 = data['u-component_of_wind_isobaric'].sel(time1=vtimes[0], isobaric=500)
vwnd_500 = data['v-component_of_wind_isobaric'].sel(time1=vtimes[0], isobaric=500)


## plotting

datacrs = ccrs.LambertConformal(
    central_latitude=data.LambertConformal_Projection.latitude_of_projection_origin,
    central_longitude=data.LambertConformal_Projection.longitude_of_central_meridian)

# A different LCC projection for the plot.
plotcrs = ccrs.LambertConformal(central_latitude=45., central_longitude=-100.,
                                standard_parallels=[30, 60])

fig = plt.figure(figsize=(17., 11.))
ax = plt.axes(projection=plotcrs)
ax.coastlines('50m', edgecolor='black')
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.set_extent([-130, -67, 20, 50], ccrs.PlateCarree())

clev500 = np.arange(5100, 6000, 60)
cs = ax.contour(x, y, ndimage.gaussian_filter(hght_500, sigma=5), clev500,
                colors='k', linewidths=2.5, linestyles='solid', transform=datacrs)
tl = plt.clabel(cs, fontsize=12, colors='k', inline=1, inline_spacing=8,
                fmt='%i', rightside_up=True, use_clabeltext=True)
# Here we put boxes around the clabels with a black boarder white facecolor
for t in tl:
    t.set_bbox({'fc': 'w'})

# Transform Vectors before plotting, then plot wind barbs.
ax.barbs(x, y, uwnd_500.data, vwnd_500.data, length=7, regrid_shape=20, transform=datacrs)

# Add some titles to make the plot readable by someone else
plt.title('500-hPa Geopotential Heights (m)', loc='left')
plt.title('VALID: %s'.format(vtimes[0]), loc='right')

plt.tight_layout()
plt.show()