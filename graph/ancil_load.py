import pandas as pd
import ancil_graph
import datetime
import monthdelta

def trading_info(tstart,filebase):
    # loads Public DVD TRADING LOAD data

    year1,month1,day1=tstart.strftime("%Y-%m-%d").split("-")
    tstart_next=tstart+monthdelta.monthdelta(1)
    year2,month2,day2=tstart_next.strftime("%Y-%m-%d").split("-")

    filename1="PUBLIC_DVD_TRADINGLOAD_{0}{1}010000.CSV".format(year1,month1)    
    filename2="PUBLIC_DVD_TRADINGLOAD_{0}{1}010000.CSV".format(year2,month2)    

    SA1=pd.read_csv(filebase+filename1)
    
    # Pivot table
    SA1=SA1.pivot_table(index="SETTLEMENTDATE",columns="DUID",values="TOTALCLEARED")
    # correct date
    SA1=SA1.set_index(pd.DatetimeIndex(SA1.index))
    SA1=SA1.loc[SA1.index.sort_values()]
    
    
    SA2=pd.read_csv(filebase+filename2)
    
    # Pivot table
    SA2=SA2.pivot_table(index="SETTLEMENTDATE",columns="DUID",values="TOTALCLEARED")
    
    SA2=SA2.set_index(pd.DatetimeIndex(SA2.index))
    SA2=SA2.loc[SA2.index.sort_values()]

    return SA1,SA2,tstart,filename1,filename2


def demand_info(tstart,filebase):
    # loads Public DVD TRADING LOAD data

    year1,month1,day1=tstart.strftime("%Y-%m-%d").split("-")
    tstart_next=tstart+monthdelta.monthdelta(1)
    year2,month2,day2=tstart_next.strftime("%Y-%m-%d").split("-")

    filename1="PUBLIC_DVD_TRADINGREGIONSUM_{0}{1}010000.CSV".format(year1,month1)    
    filename2="PUBLIC_DVD_TRADINGREGIONSUM_{0}{1}010000.CSV".format(year2,month2)    

    SA1=pd.read_csv(filebase+filename1)
    
    # Pivot table
    SA1=SA1.pivot_table(index="SETTLEMENTDATE",columns="REGIONID",values="TOTALDEMAND")
    # correct date
    SA1=SA1.set_index(pd.DatetimeIndex(SA1.index))
    SA1=SA1.loc[SA1.index.sort_values()]
    
    
    SA2=pd.read_csv(filebase+filename2)
    
    # Pivot table
    SA2=SA2.pivot_table(index="SETTLEMENTDATE",columns="REGIONID",values="TOTALDEMAND")
    
    SA2=SA2.set_index(pd.DatetimeIndex(SA2.index))
    SA2=SA2.loc[SA2.index.sort_values()]

    return SA1,SA2,tstart,filename1,filename2

def subset_pricefilter():

    return 0

def subset(df_mnt_prev,df_mnt,Gen_info,state):
    # Subsets output of trading info . E.G Looking at South Australia
    SAlist=list(Gen_info["DUID"][Gen_info['Region']==state])
    b=list(set(SAlist) & set(df_mnt.columns)& set(df_mnt_prev))
    df_mnt=df_mnt[b]
    df_mnt_prev=df_mnt_prev[b]
    return df_mnt_prev,df_mnt

def gen_df(filebase,state):
    # This function is aimed to be an entire timeslice of generators. 
    # Change to vic
    print("who goes there?")
    import datetime
    import monthdelta
    Gen_info=pd.read_excel("/mnt/y/Code/Dev/graph/MacBank/NEM Registration and Exemption List.xls",sheet_name="Generators and Scheduled Loads")

    # import pickle as pkl
    # pickle=open("/mnt/y/Data/Electricity/Average_prices/Average_prices_2013_2017.pkl")
    # prices=pkl.load(pickle)
    # prices=prices.set_index(pd.to_datetime(prices["SETTLEMENTDATE"]))
    # pickle.close()state

    tstart=datetime.datetime(2010,1,1)
    df_all=pd.DataFrame()
    for year in range(2010,2018):
      print(year)
      for month in range(1,13):
        df_mnt,df_mnt_prev,tstart,filename1,filename2=trading_info(tstart,filebase)
        #df_mnt_prev,df_mnt=subset(df_mnt_prev,df_mnt,Gen_info)
        
        mnt=pd.to_datetime(df_mnt.index.tolist())
        #df_mnt=pd.concat([df_mnt,pd.DataFrame(prices.loc[mnt][state])],axis=1)
        
        df_all=df_all.append(df_mnt)
        tstart=tstart+monthdelta.monthdelta(1)

    #df_all=df_all.set_index(pd.DatetimeIndex(df_all["SETTLEMENTDATE"]))
    #df_all=df_all.loc[df_all.index.sort_values()]
    #df_all=df_all.drop("SETTLEMENTDATE")
    df_all.to_csv("/mnt/y/Data/Electricity/Average_prices/df_all_Aust_PUBLIC_DVD_TRADING_PRICE.csv")
    # save df_all -> /mnt/y/Data/Electricity/Average_prices/df_all_VIC_PUBLIC_DVD_TRADING_PRICE

    return df_all


def SA_production_price():
    Gen_info=pd.read_excel("/mnt/y/Code/Dev/graph/MacBank/NEM Registration and Exemption List.xls",sheet_name="Generators and Scheduled Loads")
    # Read Trading prices info
    df_all=pd.read_csv("/mnt/y/Code/Analysis/graph/df_all_PUBLIC_DVD_TRADING_PRICE.csv",index_col="SETTLEMENTDATE")
    df_all=df_all.set_index(pd.DatetimeIndex(df_all.index))

    # subset stations  by wind in SA
    # Fuel source
    #Gen_info=pd.read_excel("/mnt/y/Code/Dev/graph/MacBank/NEM Registration and Exemption List.xls",sheet_name="Generators and Scheduled Loads")
    Gen_info_wind=Gen_info[Gen_info["Fuel Source - Primary"]=="Wind"]
    intersect=list(set(df_all.columns) & set(Gen_info_wind.loc[:,"DUID"]))
    price=df_all.loc[:,"SA"]
    df_all=df_all[intersect]
    return df_all,price


def subset_pricefilter(price):

    import numpy as np
   
    df_all,price=SA_production_price()
    price=price.dropna()

    p1 = np.percentile(np.array(price), 90) # return 50th percentile, e.g median.
    p2 = np.percentile(np.array(price), 95) # return 50th percentile, e.g median.

    price=price[price>p1]
    price=price[price<p2]
    times=list(price.index)

    return times

def gen_dem(filebase):

    import datetime
    Gen_info=pd.read_excel("/mnt/y/Code/Dev/graph/MacBank/NEM Registration and Exemption List.xls",sheet_name="Generators and Scheduled Loads")

    tstart=datetime.datetime(2010,1,1)
    df_all=pd.DataFrame()
    for year in range(2010,2018):
      print(year)
      for month in range(1,13):
        df_mnt,df_mnt_prev,tstart,filename1,filename2=demand_info(tstart,filebase)
        #Subset stage
        df_mnt_prev,df_mnt=df_mnt_prev["SA1"],df_mnt["SA1"]

        
        mnt=pd.to_datetime(df_mnt.index.tolist())
        
        df_all=df_all.append(df_mnt)
        tstart=tstart+monthdelta.monthdelta(1)
    return df_all

def test():
    print("thest")
    return 0

def SA_wind_func(df_all):
        # state sum
    df_all=pd.read_csv("/mnt/y/Code/Analysis/graph/df_all_PUBLIC_DVD_TRADING_PRICE.csv",index_col="SETTLEMENTDATE")
    df_all=df_all.set_index(pd.DatetimeIndex(df_all.index))
    winds=['HALLWF1',
     'SNOWNTH1',
     'HALLWF2',
     'WATERLWF',
     'LKBONNY3',
     'LKBONNY2',
     'NBHWF1',
     'CLEMGPWF',
     'SNOWTWN1',
     'SNOWSTH1',
     'BLUFF1',
     'HDWF1']
    df_all=df_all[winds]
    state_wind=df_all.sum(axis=1)
    return df_all,state_wind


def average_prices_concat():

    import datetime
    import pandas as pd
    # Real power output for comparison  For more months?
    dframe9=pd.read_excel("/mnt/y/Data/Electricity/Average_prices.xlsx",sheet_name="2009 Wind",skiprows=2)
    dframe10=pd.read_excel("/mnt/y/Data/Electricity/Average_prices.xlsx",sheetname="2010 Wind",skiprows=2)
    dframe11=pd.read_excel("/mnt/y/Data/Electricity/Average_prices.xlsx",sheetname="2011 Wind",skiprows=3)
    dframe12=pd.read_excel("/mnt/y/Data/Electricity/Average_prices.xlsx",sheetname="2012 Wind",skiprows=3)
    dframe13=pd.read_excel("/mnt/y/Data/Electricity/Average_prices.xlsx",sheetname="2013 Wind",skiprows=3)
    dframe14=pd.read_excel("/mnt/y/Data/Electricity/Average_prices.xlsx",sheetname="2014 Wind",skiprows=3)
    dframe15=pd.read_excel("/mnt/y/Data/Electricity/Average_prices.xlsx",sheetname="2015 Wind",skiprows=3)
    dframe16=pd.read_excel("/mnt/y/Data/Electricity/Average_prices.xlsx",sheetname="2016 Wind",skiprows=3)
    dframe17=pd.read_excel("/mnt/y/Data/Electricity/Average_prices.xlsx",sheetname="2017 Wind",skiprows=3)

    dframe=pd.concat([dframe13,dframe14,dframe15,dframe16,dframe17])


    dframe = dframe[dframe['SETTLEMENTDATE'].apply(lambda x: type(x) == datetime.datetime)]

    dframe.index = dframe['SETTLEMENTDATE']

    #dframe.to_pickle("/mnt/y/Code/wind2power/windvalue/RRP_prices_real_2009_2017.pkl",index="SETTLEMENTDATE")
    dframe.to_pickle("/mnt/y/Data/Electricity/Average_prices/Average_prices_2009_2017.pkl",index="SETTLEMENTDATE")
    return 0 


def rural_lga():

    import geopandas
    import numpy as np

    # this shapefile is from natural earth data
    # http://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/
    states = geopandas.read_file('/mnt/y/Code/Dev/Untitled Folder/LGA_POLYGON')

    # Subset shires and cities
    city=[s for s in states["LGA_OFFICI"] if s.split(" ")[-1]=="CITY"]
    shire=[s for s in states["LGA_OFFICI"] if s.split(" ")[-1]=="SHIRE"]

    # bounding box over melbounre
    lon1=states[states["LGA_OFFICI"].isin(city)].centroid.x<145.5
    lon2=states[states["LGA_OFFICI"].isin(city)].centroid.x>144.0
    lat1=states[states["LGA_OFFICI"].isin(city)].centroid.y<-37.0

    # inner city regions
    innercity=np.array(lon1)&np.array(lon2)&np.array(lat1)

    #exclude inner city regions
    outercity=np.array(city)[list(np.logical_not(innercity))]

    #combine shires and outer city regions
    rural_lga=np.append(outercity,np.array(shire))
    return rural_lga

def load_Bret_price(Bret_csv):
    # Future Price
    # "/mnt/y/Data/Electricity/future/Half-hour output prices.xlsx"
    future=pd.read_excel(Bret_csv,skiprows=2)

    # To store all future prices
    price_future=pd.DataFrame()
    start=dt.datetime(2019,1,1,0,30)


    for year in range(2019,2030):
        price_year=pd.DataFrame()
        price_year=pd.DataFrame(future[year])
        price_year=price_year[price_year!=0]
        price_year=price_year.dropna()
        index=pd.date_range(dt.datetime(year,1,1,0,30),periods=len(price_year),freq='30min')
        price_year.index=index
        #price_future=price_future.append(price_year)
        price_year.columns=["$/MW"]
        price_future=pd.concat([price_future,price_year],axis=0)
    return 0

def conversion_chart():
    # convert wind farm coordaintes in project pipeline to LGA3 and LGA2

    import pandas as pd
    import geopandas
    # Convert from wind farm coordinates to LGA3 to LGA2
    lga_converter=pd.read_csv("/mnt/y/concepts/LGA_area/lga_1_3_complete.csv")

    # Read in wind farm coordinates
    wind_coordinates=pd.read_excel("/mnt/y/Data/Electricity/RepuTex wind and solar project pipeline_All states_July 2018 v3.xlsx",skiprows=4)


    import shapely
    # look just at vic wind farms
    states = geopandas.read_file('/mnt/y/Code/Dev/Untitled Folder/LGA_POLYGON')

    df_project=list()
    df_lga2=list()
    df_lga3=list()

    wind_coordinates=wind_coordinates[wind_coordinates["Type"]=="Wind "]
    wind_coordinates=wind_coordinates[wind_coordinates["State "]=="Victoria "]
    # loop over wind farm
    for project in wind_coordinates["Project"]:
        lat=wind_coordinates[wind_coordinates["Project"]==project]["Latitude"]
        lon=wind_coordinates[wind_coordinates["Project"]==project]["Longitude"]
        p1=shapely.geometry.Point(lon,lat)
        
        # find overlap LGA from states
        LGA3=states[states['geometry'].intersects(p1)]["LGA_NAME"].iloc[0]
        # convert to LGA2
        mask=lga_converter["LGA3"].apply(lambda x: LGA3 in x)
        LGA2=lga_converter[mask]["LGA2"]
        df_project.append(project)
        df_lga2.append(LGA2.iloc[0])
        df_lga3.append(LGA3)
        
    df_proj_LGA=pd.DataFrame({"Project":pd.Series(df_project),"LGA2":pd.Series(df_lga2),"LGA3":pd.Series(df_lga3)})
        
   

    #df_proj_LGA["LGA2"].index=correlation_table.index
    #correlation_table["LGA2"]=df_proj_LGA["LGA2"]
    #df_proj_LGA.to_csv("/mnt/y/concepts/LGA_area/helpful_conversion_chart.csv")
    return df_proj_LGA


### Images ###

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def powerstations_geo():

    import pandas as pd
    ps=pd.read_csv("/mnt/y/Data/Electricity/Generation Trading load/MajorPowerStations_v2.csv")
    return ps

def Cartopy_Earth_temp(tm,var,rrp,basefn):

    import matplotlib
    matplotlib.use('Agg') #https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable


    from netCDF4 import Dataset, MFDataset, num2date
    import matplotlib.pylab as plt
    import numpy as np
    from matplotlib import cm
    import cartopy.crs as ccrs
    import pandas as pd
    import sys
    import os
    from cartopy.util import add_cyclic_point

    ## Temperature
    # Load data
    #http://earthpy.org/cartopy_backgroung.html

    year  =  tm.year
    month =  tm.month
    day   =  tm.day
    hour  =  tm.hour
    
    tind = (day-1)*24+hour
    basefn=basefn.format(year,month)

    base = "/mnt/y/Data/Weather/ECMWF/TEST/Saved_plots/{1}/{2:02d}/{0}/".format(var,year,month)
    mkdir_p(base)

    flf = Dataset(basefn)
    lat = flf.variables['latitude'][:]
    lon = flf.variables['longitude'][:]


    temp = flf.variables[var][tind,:,:]

    crs_latlon=ccrs.PlateCarree()
    plt.figure(figsize=(13,6.2))
        
    ax = plt.subplot(111, projection=crs_latlon)

    mm = ax.pcolormesh(lon,\
                       lat,\
                       temp,\
                       vmin=273,\
                       vmax=310,\
                       transform=ccrs.PlateCarree(),cmap="RdBu_r" )

    ax.coastlines();
    plt.colorbar(mm)

    ## add point
    ps=pd.read_csv("/mnt/y/Data/Electricity/Generation Trading load/MajorPowerStations_v2.csv")
    ps_wind=ps[ps["GENERATIONTYPE"]=="Wind Turbine"]


    for i,p in enumerate(ps.index):
        LAT=ps.iloc[i,:]["LATITUDE"]
        LON=ps.iloc[i,:]["LONGITUDE"]
        plt.plot([LON],[LAT],color='red',marker="o",markersize=4)#,transform=crs_latlon)
        
    for i,p in enumerate(ps_wind.index):
        LAT=ps_wind.iloc[i,:]["LATITUDE"]
        LON=ps_wind.iloc[i,:]["LONGITUDE"]
        plt.plot([LON],[LAT],color='green',marker="o",markersize=4,)#,transform=crs_latlon)


    # SA bounary coordinates
    ax.set_extent((125, 141, -25, -39), crs=crs_latlon)
    plt.savefig(base+"era5_AUS_{0}_{1}{2:02d}{3:02d}_{4:02d}__rrp{5}.png".format(var,year,month,day,hour,rrp))

    return 0



def Cartopy_Earth_wind(tm,var,rrp,basefn):

    import datetime as dt
    tm=dt.datetime(2010,1,1,0,1)
    rrp = "wind"
    basefn="/mnt/y/Data/Weather/ECMWF/Data_R/ml/{0}/{1:02d}/era5_hourly_ml_AUS_{0}{1:02d}.nc"


    import matplotlib
    matplotlib.use('Agg') #https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable

    from netCDF4 import Dataset, MFDataset, num2date
    import matplotlib.pylab as plt
    import numpy as np
    from matplotlib import cm
    import cartopy.crs as ccrs
    import pandas as pd
    import sys
    import os
    from cartopy.util import add_cyclic_point

    ## Wind

    year  =  tm.year
    month =  tm.month
    day   =  tm.day
    hour  =  tm.hour

    tind = (day-1)*24+hour
    basefn=basefn.format(year,month)

    base = "/mnt/y/Data/Weather/ECMWF/TEST/Saved_plots/{1}/{2:02d}/{0}/".format(var,year,month)
    ancil_load.mkdir_p(base)

    flf = Dataset(basefn)
    lat = flf.variables['latitude'][:]
    lon = flf.variables['longitude'][:]

    u = flf.variables["u100"][tind,:,:]
    v = flf.variables["v100"][tind,:,:]


    crs_latlon=ccrs.PlateCarree()
    plt.figure(figsize=(13,6.2))

    ax = plt.subplot(111, projection=crs_latlon)

    mm = ax.pcolormesh(lon,\
                       lat,\
                       temp,\
                       vmin=273,\
                       vmax=310,\
                       transform=ccrs.PlateCarree(),cmap="RdBu_r" )
    plt.colorbar(mm)
    ax.coastlines();

        #ax.quiver(lon,lat,u,v,transform=crs_latlon, headwidth=1, scale =1.0 headlength=4)
    skip=(slice(None,None,5),slice(None,None,5))
    ax.quiver(lon[::5],lat[::5],u[skip],v[skip],color="pink",transform=crs_latlon, headwidth=2, headlength=3)



    ## add point
    ps=pd.read_csv("/mnt/y/Data/Electricity/Generation Trading load/MajorPowerStations_v2.csv")
    ps_wind=ps[ps["GENERATIONTYPE"]=="Wind Turbine"]


    for i,p in enumerate(ps.index):
        LAT=ps.iloc[i,:]["LATITUDE"]
        LON=ps.iloc[i,:]["LONGITUDE"]
        plt.plot([LON],[LAT],color='red',marker="o",markersize=4)#,transform=crs_latlon)

    for i,p in enumerate(ps_wind.index):
        LAT=ps_wind.iloc[i,:]["LATITUDE"]
        LON=ps_wind.iloc[i,:]["LONGITUDE"]
        plt.plot([LON],[LAT],color='green',marker="o",markersize=4,)#,transform=crs_latlon)


    # SA bounary coordinates
    #ax.set_extent((125, 141, -25, -39), crs=crs_latlon)
    plt.savefig(base+"WIND.png".format(var,year,month,day,hour,rrp))

