# Working in jupyter notebook
# does it work in ipython

# easier earhtpy
#http://earthpy.org/tag/cartopy.html

# at this point we can change projections etc. 

def Cartopy_Earth_temp(tm,tind,var)

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

	year  =  tm.index.year
    month =  tm.index.month
    day   =  tm.index.day
    hour  =  tm.index.hour
    
    tind = (day-1)*24+hour
    basefn="/mnt/y/Data/Weather/ECMWF/Data_R/ml/{0}/{1}/era5_hourly_ml_AUS_{0}{1}.nc".format(year,month)

	flf = Dataset(filename)
	lat = flf.variables['latitude'][:]
	lon = flf.variables['longitude'][:]


	temp = flf.variables[var][tind,:,:]



	plt.figure(figsize=(13,6.2))
	    
	ax = plt.subplot(111, projection=ccrs.PlateCarree())

	mm = ax.pcolormesh(lon,\
					   lat,\
	                   temp,\
	                   vmin=273,\
	                   vmax=310,\
	                   transform=ccrs.PlateCarree(),cmap="RdBu_r" )

	ax.coastlines();


	# TODO add WWIND (and solar?) next to this. 


	plt.savefig("/mnt/y/Data/Weather/ECMWF/Saved_plots/era5_AUS_{0}_{1}{2:02d}{3:02d}_{4:02d}".format(var,year,month,day,hour))


