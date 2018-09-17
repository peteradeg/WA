# ancil_analytics is the place for  iteratively looking at PCA



def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None,title="test",ytitle="y"):
	import matplotlib.pyplot as plt
	# plot the shaded range of the confidence intervals
	plt.fill_between(range(mean.shape[0]), ub, lb,color=color_shading, alpha=.5)
	# plot the mean on top
	plt.plot(mean, color_mean)
	plt.title(title)
	plt.ylabel(ytitle)


def timeofday_plot(df_diurnal,df_diurnal_std1,df_diurnal_std2,regions):

	# Plotting seaborn facet.
	import matplotlib.pyplot as plt

	#quarter=["Q1","Q2","Q3","Q4"]

	# plot price
	#priceVic

	# plot regions:
	# for i,s in enumerate(["summer","autumn","winter","spring"]):
	# #for i,s in enumerate(["Q1","Q2","Q3","Q4"]):
	#     plt.subplot(5,4,i+1)
	#     plot_mean_and_CI(price_dict[s], price_dict_std1[s], price_dict_std2[s], color_mean='r', color_shading='k',title=s,ytitle="Price")


	season_list=["summer","autumn","winter","spring"]
	for i in range(0,4):
	    #j is season
	    for j in range(0,4):
	        # i is region
	        plt.subplot(5,4,j*4+(i+1))
	        df_diurnal_season  =  df_diurnal[df_diurnal["season"]==season_list[i]][regions[j]]
	        df_diurnal_season  =  df_diurnal_season.reset_index()
	        df_diurnal_season  =  df_diurnal_season.drop(["index"],axis=1)
	        df_diurnal_std1_season  =  df_diurnal_std1[df_diurnal_std1["season"]==season_list[i]][regions[j]]
	    #    df_diurnal_std1_season  = df_diurnal_std1_season.reset_index()
	        df_diurnal_std2_season  =  df_diurnal_std2[df_diurnal_std2["season"]==season_list[i]][regions[j]]
	    #    df_diurnal_std2_season  = df_diurnal_std2_season.reset_index()

	        plot_mean_and_CI(df_diurnal_season, df_diurnal_std1_season, df_diurnal_std2_season, color_mean='r', color_shading='k',title=season_list[i],ytitle=regions[j])


	return 0

def timeofday_production(df_generation):
	import pandas as pd
	import numpy as np
	import seaborn as sns
	# df_generation = is hourly production data like df_all


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

# # # Price later
#priceVic=pd.read_csv("/mnt/y/Data/Electricity/Average_prices_2009_2017.csv")
#priceVic=priceVic[priceVic!=0]
#priceVic=priceVic.dropna()

def timeofday_price():

	# price data
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






