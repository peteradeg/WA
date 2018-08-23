# ancil timeofday analysis

import pandas as pd
import numpy as np
import seaborn as sns



def lilly2diurnal():

	df_generation = "/mnt/y/Data/Power/ECMWF/lillypond/future_wind.csv"

	# find current wind generation DUID Dispatch
	#"/mnt/y/Data/Electricity/future/"

	df_generation = pd.read_csv(df_generation)
	df_generation.index=pd.DatetimeIndex(df_generation["time"])
	df_generation=df_generation.drop(["time","Unnamed: 0"],axis=1)

	df_diurnal=pd.DataFrame()
	df_diurnal_std1=pd.DataFrame()
	df_diurnal_std2=pd.DataFrame()

	df_diurnal_P=pd.DataFrame()
	df_diurnal_std1_P=pd.DataFrame()
	df_diurnal_std2_P=pd.DataFrame()

	df_mean=pd.DataFrame()
	df_std1=pd.DataFrame()
	df_std2=pd.DataFrame()

	seasonlist= {"summer":[12,1,2],"autumn":[3,4,5],"winter":[6,7,8],"spring":[9,10,11]}
	#seasonlist= {"summer":[1,2,3],"autumn":[4,5,6],"winter":[7,8,9],"spring":[10,11,12]}


	for season in ["summer","autumn","winter","spring"]:
	    season_list=seasonlist[season]
	    df_season=df_generation[np.logical_or.reduce((df_generation.index.month==season_list[0],df_generation.index.month==season_list[1],df_generation.index.month==season_list[2]))]
	    #price
	    for hour in range(0,24):
	        
	        df_hour=df_season[df_season.index.hour==hour]
	        
	        for site in df_hour.columns:
	            mean=df_hour[site].mean()
	            df_mean[site]=pd.Series(mean)
	            df_std1[site]=pd.Series(df_hour[site][df_hour[site]>mean].std())+mean
	            df_std2[site]=(pd.Series(df_hour[site][df_hour[site]<mean].std())-mean)*-1

	        df_mean["hour"]=hour
	        df_mean["season"]=season
	        
	        df_std1["hour"]=hour
	        df_std1["season"]=season
	        
	        df_std2["hour"]=hour
	        df_std2["season"]=season
	            
	        df_diurnal=df_diurnal.append(df_mean)
	        df_diurnal_std1=df_diurnal_std1.append(df_std1)
	        df_diurnal_std2=df_diurnal_std2.append(df_std2)
	    
	        
	df_diurnal=df_diurnal.reset_index()
	df_diurnal_std1=df_diurnal_std1.reset_index()
	df_diurnal_std2=df_diurnal_std2.reset_index()

	return df_diurnal,df_diurnal_std1,df_diurnal_std2

def price2diurnal():

	price_dict={"summer":[],"autumn":[],"winter":[],"spring":[]}
	price_dict_std1={"summer":[],"autumn":[],"winter":[],"spring":[]}
	price_dict_std2={"summer":[],"autumn":[],"winter":[],"spring":[]}
	seasonlist= {"summer":[12,1,2],"autumn":[3,4,5],"winter":[6,7,8],"spring":[9,10,11]}

	for season in ["autumn","summer","winter","spring"]:
	    season_list=seasonlist[season]
	    price_hour=[]
	    price_std1=[]
	    price_std2=[]
	    df_season_P=priceVic[np.logical_or(priceVic.index.month==season_list[0],priceVic.index.month==season_list[1],priceVic.index.month==season_list[2])]
	    for hour in range(0,24):
	        df_hour_P=df_season_P[df_season_P.index.hour==hour]
	        mean_price = df_hour_P.mean()
	        std1       = df_hour_P[df_hour_P>mean_price].std()+mean_price
	        std2       = (df_hour_P[df_hour_P<mean_price].std()-mean_price)*-1
	        
	        price_hour.append(mean_price)
	        price_std1.append(std1)
	        price_std2.append(std2)
	    price_dict[season]=np.array(price_hour)
	    price_dict_std1[season]=np.array(price_std1)
	    price_dict_std2[season]=np.array(price_std2)

	return price_dict,price_dict_std1,price_dict_std2

