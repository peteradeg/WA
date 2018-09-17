#Graphing ancil

import pandas as pd
import datetime
import networkx as nx
import ancil_load




def valuefactor(df_delta,site,price):
    #todo time wrangling above?
    #price=df_delta["SA"]
    price=price.loc[df_delta.index]
    # multiply site and price
    mysite=(df_delta[site]*price*.5).sum()/(df_delta[site].sum()*.5)
    # multiply state and price
    state=df_delta.sum(axis=1)
    mystate=(state*price*.5).sum()/(state.sum()*.5)
    # VF as percentage
    vf=mysite*100/mystate

    return vf

def cbn1(G):

    cbn={}
    for node in G.nodes():
        g_cbn=0
        for edge in G[node]:
            g_cbn=g_cbn+G[node][edge]["weight"]
        cbn[node]=g_cbn/len(G[node])

    nx.set_node_attributes(G,name="CBN1",values=cbn)

    return G


def cbn2(G):
    import numpy as np

    node_power=nx.get_node_attributes(G,"nvalue")

    cbn={}
    for node in G.nodes():
        g_cbn=0
        for edge in G[node]:
            g_cbn=g_cbn+G[node][edge]["weight"]*node_power[node]

        tpower=np.array(node_power.values()).sum()-node_power[node]
        cbn[node]=g_cbn/(len(G[node])*tpower)

    nx.set_node_attributes(G,name="CBN2",values=cbn)
    return G



def GfromAEMO(df_mnt):
    import math
    df=df_mnt.corr() # todo change from corr
    del df.index.name
    del df.columns.name

    mst_links = df.stack().reset_index()
    mst_links=mst_links.loc[ (abs(mst_links[0]) != 0) & (mst_links['level_0'] != mst_links['level_1'])]

    mst_links=mst_links.rename(columns={0:"weight"})
    G=nx.from_pandas_edgelist(mst_links, 'level_0', 'level_1',edge_attr="weight")

    for e in G.edges():
      try:
        # TODO change from Corr
        #G[e[0]][e[1]]["weight"]=G[e[0]][e[1]]["weight"]/(abs(mst_links["weight"]).max())
        #G[e[0]][e[1]]["weight"]=abs(2-math.sqrt(2*(1-G[e[0]][e[1]]["weight"])))
        G[e[0]][e[1]]["weight_length"]=math.sqrt(2*(1-G[e[0]][e[1]]["weight"]))
      except ValueError:
        G[e[0]][e[1]]["weight"]=0
    return G



def attributes_two(G,dataframe_matrix,before_dataframe_matrix):

    # include attribute
    # Ways to size month value nodes
    nsize_now={}
    for n in G.nodes:
        nsize_now[n]=sum(pd.to_numeric(dataframe_matrix[n]))
    
    # Ways to size month delta value nodes
    nsize_before={}
    for n in G.nodes:
        nsize_before[n]=sum(pd.to_numeric(before_dataframe_matrix[n]))


    # factor change vn+1/vn -1
    nfactorchange={}
    for n in G.nodes:
        nfactorchange[n]=(nsize_now[n]/nsize_before[n])-1

        
    # for delta loop    
    ndelta={}
    absndelta={}
    for n in G.nodes:
        ndelta[n]=round(nsize_now[n]-nsize_before[n],0)
        absndelta[n]=abs(round(nsize_now[n]-nsize_before[n],0))

    # give absweight attribute
    w=nx.get_edge_attributes(G,"weight")
    for key in w.keys():
        w[key]=abs(float(w[key]))
    nx.set_edge_attributes(G,name="absweight",values=w)
    # Define other attributes
    #TODO include Bin (Vi/vi+1 -1)
    nx.set_node_attributes(G,name="ndelta",values=ndelta)
    nx.set_node_attributes(G,name="absndelta",values=absndelta)
    nx.set_node_attributes(G,name='nfactorchange',values=nfactorchange)
    


    return G,ndelta 

def attributes_one(G,dataframe_matrix,dfr_1,price):

    nsize_now={}
    for n in G.nodes:
        #nsize_now[n]=sum(pd.to_numeric(dataframe_matrix[n]))
        nsize_now[n]=dataframe_matrix[n].mean() # TODO changed from max to mean
    nx.set_node_attributes(G,name="nvalue",values=nsize_now)

    nsize_price={}
    for n in G.nodes:
        #nsize_now[n]=sum(pd.to_numeric(dataframe_matrix[n]))
        nsize_price[n]=dfr_1[n].sum()
    nx.set_node_attributes(G,name="nrevenue",values=nsize_price)

    # give absweight attribute
    w=nx.get_edge_attributes(G,"weight")
    for key in w.keys():
        w[key]=abs(float(w[key]))
    nx.set_edge_attributes(G,name="absweight",values=w)


    # Value Factor
    # load in price


    nvaluefactor={}
    for n in G.nodes:
        nvaluefactor[n]=valuefactor(dfr_1,n,price)
    nx.set_node_attributes(G,name="nvaluefactor",values=nvaluefactor)



    return G



def attributes_color(G):
    import pandas as pd
    import seaborn as sns
    import numpy as np
    # Colour edges
    esign={}
    for e in G.edges:
        if G[e[0]][e[1]]["weight"] < 0:
          esign[e]="red"
        else:
          esign[e]="green"

    # add in colour relative to VF
    #palette=sns.color_palette("RdBu_r",11)[::-1]
    palette=sns.diverging_palette(10,150, n=12)
    palette=palette[:4]+palette[5:]
    #palette[5]=np.array([255,141,89,50])/255.0

    vfbins=np.array([50,60,70,80,90,100,110,120,130,140,150])

    nsign={}
    for n in G.nodes():
        idx=(np.abs(vfbins-nx.get_node_attributes(G,"nvaluefactor")[n])).argmin()
        nsign[n]=palette[idx]

    # ADD in colour relative to regions LGA1_3
    #read in LGA1_3
    # lga_area=pd.read_csv("/mnt/y/concepts/LGA_area/lga_1_3_complete.csv")
    # lga_area.index=lga_area["LGA3"]
    # cpalatte={"North West":'#d73027',"Central Vic":'#fc8d59',"South West":'#fee090',"Northern Vic":'#e0f3f8',"Gippsland":'#91bfdb'}
    # nsign={}
    # for n in G.nodes:
    #     nsign[n]=cpalatte[lga_area.loc[n]["LGA1"]]

    # Colour nodes for tow attributes
    # nsign={}
    # for e in G.nodes:
    #     if nx.get_node_attributes(G,"ndelta")[e] < 0:
    #       nsign[e]="red"
    #     else:
    #       nsign[e]="green"

    nx.set_edge_attributes(G,name="esign",values=esign)
    nx.set_node_attributes(G,name="nsign",values=nsign)
    return G

def attributes_DUID(G,DUID):

	DRUID=(DUID["Fuel Source - Primary"]).sort_values().unique()
	color_book={"Battery storage":"red","Fossil":"brown","Hydro":"blue",'Renewable/ Biomass / Waste':"pink","Solar":"yellow","Wind":"green","nan":"black"}
	generator_type={}
	generator_state={}
	generator_secondary={}

	AEMO_fail={}
	for node in G.nodes():

		if len(DUID[DUID["DUID"]==node])!=0:
			#print(node)
			out=DUID[DUID["DUID"]==node]["Fuel Source - Primary"]
			generator_type[node]=color_book["{0}".format(out.unique()[0])]

			out=DUID[DUID["DUID"]==node]["Fuel Source - Descriptor"]
			generator_secondary[node]=out.unique()[0]

			out=DUID[DUID["DUID"]==node]["Fuel Source - Primary"]
			generator_state[node]=out.unique()[0]
		else:
			AEMO_fail[node]="orange"
			generator_color[node]="orange"

	nx.set_node_attributes(G,name="Generator type",values=generator_type)
	nx.set_node_attributes(G,name="Generator type secondary",values=generator_secondary)
	nx.set_node_attributes(G,name="Generator state",values=generator_state)

	return G


def attributes(G):
    #elen={}
    #for e in G.edges:
    ## Centrality metrics
    G_clustering=nx.clustering(G)
    G_deg=nx.degree_centrality(G)
    G_degree=nx.degree(G)
    #G_bet=nx.betweenness_centrality(G)
    #G_eig=nx.eigenvector_centrality_numpy(G)
    #G_page=nx.pagerank_numpy(G)
    #G_load=nx.load_centrality(G)
    #G_katz=nx.katz_centrality_numpy(G)
    G_closeness=nx.closeness_centrality(G) # aka node strenght https://arxiv.org/pdf/0803.3884.pdf
    # closeness 
    #print(G_closeness)
    #Centrality_metric={"Degree_centrality":G_deg,"Betweeness":G_bet,"Eigencentrality":G_eig,"load":G_load,"katz":G_katz,"Pagerank":G_page,"Closeness":G_closeness,"Clustering":G_clustering}
    Centrality_metric={"Degree_centrality":G_deg,"Clustering":G_clustering}
    for cent in Centrality_metric:
        nx.set_node_attributes(G,name=cent,values=Centrality_metric[cent])

    d = {key: value for (key, value) in G_degree}
    nx.set_node_attributes(G,name="Degree",values=d)

   
    #CBN1.0 Centrality by node 
    G=cbn1(G)
    G=cbn2(G)

        
    #return G,G_page,G_katz,G_closeness
    return G


# TODO
# def correlation_bygraph()
# total correlation of graph
# avergae not (normalise)


#TODO blobs
# time to include this.

def attributes_time(G):
    # Issues with csv 
    # option is to look at XML
	return G

# not working not sure why?
def powertovalue(df_mnt,prices):
    # convert df_mnt_prev
    mnt=pd.to_datetime(df_mnt.index.tolist())
    #col=df_mnt.columns[0]
    df_mnt_out=pd.DataFrame()
    for col in df_mnt.columns:
      df_mnt_out[col]=df_mnt[col]*prices.loc[mnt,"SA"]
    return df_mnt_out

def makeG(df_size,df_attr,year,price):
    # df_size relates to size of node for example it could be power price
    # df_attr relates to the power output
    import numpy as np
    #G=GfromAEMO(df_size) #23/07 
    G=GfromAEMO(df_attr)
    G=nx.maximum_spanning_tree(G)
    G=attributes_one(G,df_size.replace([np.inf, -np.inf], np.nan),df_attr,price)
    G=attributes(G)
    G=attributes_color(G)

    # at some point will need simple plot TODO
    graphout=graph_build_lga(G,year,"/mnt/y/Code/Analysis/graph/Graph_figures/VIC_LGA/",labels=True)
    return G



### graph build ###
def graph_build(G,mnt,sname_base,labels=False):
    # Build your graph
    import matplotlib.pyplot as plt
    import numpy as np
    from decimal import Decimal
    
    # Plot the network:
    f=plt.figure(figsize=(20,10))
    pos=nx.spring_layout(G,k=.2,iterations=100,weight="absweight")
    
    nsize=[]
    for k in G.nodes():
        #nsize.append(G.nodes[k]["absndelta"])
        nsize.append(G.nodes[k]["nvalue"])
        #nsize.append(G.nodes[k]["Eigencentrality"])
        
    
    # make list for plotting
    elabels = nx.get_edge_attributes(G, "weight")
    

    nlabels={}
    #print(nlabels)
    ncolor=[]
    for k in G.nodes():
        #print(k,"----",G.nodes[k]["Pagerank"])
        if labels:
          # Make the label "pretty" 
          
          nlabels[k]="{0} {1:2.0f} /n {0} {2:2.0f} ".format(k,G.nodes[k]["nvalue"])
        else:
          nlabels[k]=k
        #nlabels[k]="{0} {1:4.0f} W".format(k,nsize[k])
        #nlabels[k]="{0} {1:4.0f} W".format(k,G.nodes[k]["Eigencentrality"])
        ncolor.append(G.nodes[k]["nsign"])
        #ncolor.append(G.nodes[k]["Generator type"])
        #ncolor.append(G.nodes[k]["Generator state"])
    ecolor=[]
    for k in G.edges():
        #elabels.append("{:1.2f}".format(G[k[0]][k[1]]['weight']))
        elabels[k]="{:1.2f}".format(G[k[0]][k[1]]['weight'])
        ecolor.append(G.edges()[k]["esign"])
    edges = G.edges()
 
    
    sname=sname_base+"{0}.png".format(mnt)
    #print(nlabels)
    nsize=np.array(nsize)*2000/(np.array(nsize).max())
    
    nx.draw(G, pos, edges=edges,font_size=12,node_size=nsize,labels=nlabels,edge_color=ecolor,node_color=ncolor)
    print("hi")
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=elabels)
    #plt.legend(numpoints = 1)
    f.savefig(sname)
    return ecolor,ncolor,nlabels,elabels,nsize

def graph_build_lga(G,mnt,sname_base,labels=False):
    # Build your graph
    import matplotlib.pyplot as plt
    import numpy as np
    from decimal import Decimal
    
    # Plot the network:
    f=plt.figure(figsize=(20,10))
    pos=nx.spring_layout(G,k=.2,iterations=100,weight="weight_length")
    
    nsize=[]
    for k in G.nodes():
        nsize.append(G.nodes[k]["nrevenue"])
       
    
    # make list for plotting
    elabels = nx.get_edge_attributes(G, "weight")
    

    nlabels={}
    #print(nlabels)
    ncolor=[]
    for k in G.nodes():
        #print(k,"----",G.nodes[k]["Pagerank"])
        if labels:
          #nlabels[k]="{0} {1:4.0f}".format(k,G.nodes[k]["ndelta"])
          k_star=" ".join(k.split(" ")[:-1])
          #nlabels[k]="{0} {1:.1E} MW  {2:.1E} $ ".format(k_star,G.nodes[k]["nvalue"],G.nodes[k]["nrevenue"])
          #nlabels[k]="{0} {1:.1E} MW  {2:.1E} $ ".format(k_star,float("%.1g" % (G.nodes[k]["nvalue"]*10**-9)),float("%.1g" % (G.nodes[k]["nrevenue"]*10**-9)))
          nlabels[k]="{0}".format(k_star)
        else:
          nlabels[k]=k
        #nlabels[k]="{0} {1:4.0f} W".format(k,nsize[k])
        #nlabels[k]="{0} {1:4.0f} W".format(k,G.nodes[k]["Eigencentrality"])
        ncolor.append(G.nodes[k]["nsign"])
        #ncolor.append(G.nodes[k]["Generator type"])
        #ncolor.append(G.nodes[k]["Generator state"])
    ecolor=[]
    for k in G.edges():
        #elabels.append("{:1.2f}".format(G[k[0]][k[1]]['weight']))
        elabels[k]="{:1.2f}".format(G[k[0]][k[1]]['weight'])
        ecolor.append(G.edges()[k]["esign"])
    edges = G.edges()
 
    
    sname=sname_base+"{0}.png".format(mnt)
    #print(nlabels)
    # emphasize upper end
    nsize=np.array(nsize)*2000/(np.array(nsize).max())
    upper=nsize[nsize>.8*nsize.max()]
    amx=upper.max()
    amn=upper.min()
    nsize[nsize>.8*nsize.max()]=(upper-amn)*1000/(amx-amn)+np.sort(upper)[0]



    #p50=nsize>.5*nsize.max()
    #nsize[p50]=np.range(nsize[p50].min(),nsize[p50])

    #nsize=np.array(nsize)*2000/(np.array(nsize).max())
    
    nx.draw(G, pos, edges=edges,font_size=12,node_size=nsize,labels=nlabels,edge_color=ecolor,node_color=ncolor)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=elabels)
    plt.legend(numpoints = 1)
 

    f.savefig(sname)
    return ecolor,ncolor,nlabels,elabels,nsize