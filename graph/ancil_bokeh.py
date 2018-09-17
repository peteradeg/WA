# Bokeh Plotting


def convert_GeoPandas_to_Bokeh_format(gdf):
    """
    Function to convert a GeoPandas GeoDataFrame to a Bokeh
    ColumnDataSource object.
    
    :param: (GeoDataFrame) gdf: GeoPandas GeoDataFrame with polygon(s) under
                                the column name 'geometry.'
                                
    :return: ColumnDataSource for Bokeh.
    """
    gdf_new = gdf.drop('geometry', axis=1).copy()
    gdf_new['x'] = gdf.apply(getGeometryCoords, 
                             geom='geometry', 
                             coord_type='x', 
                             shape_type='polygon', 
                             axis=1)
    
    gdf_new['y'] = gdf.apply(getGeometryCoords, 
                             geom='geometry', 
                             coord_type='y', 
                             shape_type='polygon', 
                             axis=1)
    
    return ColumnDataSource(gdf_new)

def getGeometryCoords(row, geom, coord_type, shape_type):
    """
    Returns the coordinates ('x' or 'y') of edges of a Polygon exterior.
    
    :param: (GeoPandas Series) row : The row of each of the GeoPandas DataFrame.
    :param: (str) geom : The column name.
    :param: (str) coord_type : Whether it's 'x' or 'y' coordinate.
    :param: (str) shape_type
    """
    
    # Parse the exterior of the coordinate
    if shape_type == 'polygon':
        exterior = row[geom].exterior
        if coord_type == 'x':
            # Get the x coordinates of the exterior
            return list( exterior.coords.xy[0] )    
        
        elif coord_type == 'y':
            # Get the y coordinates of the exterior
            return list( exterior.coords.xy[1] )

    elif shape_type == 'point':
        exterior = row[geom]
    
        if coord_type == 'x':
            # Get the x coordinates of the exterior
            return  exterior.coords.xy[0][0] 

        elif coord_type == 'y':
            # Get the y coordinates of the exterior
            return  exterior.coords.xy[1][0]


def bokeh_PCA(geolocate,PCX,title="Eigenvector",):

	from bokeh.plotting import figure, save, output_file
	from bokeh.models import ColumnDataSource
	from bokeh.models import HoverTool
	from bokeh.palettes import Viridis256, RdYlBu

	import seaborn as sns
	import bokeh.models

	my_hover = HoverTool()
	my_hover.tooltips = [("Wind Farm",'@{Wind Farm}'),("Location","@Location"),("Construction Year","@{Year entered}"),("PC{0}".format(PCX),"@PC{0}".format(PCX))]


	tab = figure(title=title)

	# ADDING BASEMAP
	#p.multi_line('x',"y",source=base_chi,color="black",line_width=2)
	#base.plot()

	p_df = geolocate.drop('Coordinates', axis=1).copy()
	psource = ColumnDataSource(p_df)

	# Delux
	palette=sns.diverging_palette(10,150, n=31)
	palette=palette.as_hex()

	#mapper=bokeh.models.mappers.ContinuousColorMapper(palette=Viridis256)
	mapper=bokeh.models.mappers.LinearColorMapper(palette=Viridis256[ :: -1], low=-1, high=1)
	colors= { 'field': 'PC{0}'.format(PCX), 'transform': mapper}
	# Quick test on radius
	rads= { 'field': 'PC{0}'.format(PCX), 'transform': mapper}


	#p.circle('Latitude','Longitude',source=psource , fill_color= "red",size=12,radius=radii)

	# radii = locate["PC1"]*.5
	# p.circle(x=p_df['Latitude'],y=p_df['Longitude'] , fill_color= palette[0:p_df.shape[0]],size=12,radius=radii)
	tab.circle(y='Latitude',x='Longitude',source=psource , fill_color=colors,size=1,radius="PC{0}_r".format(PCX),fill_alpha=.3)
	#tab.circle(y='Latitude',x='Longitude',source=psource , fill_color=colors,size=1,radius="PC{0}_r".format(PCX))

	tab.add_tools(my_hover)

	outfp="/mnt/y/Code/Analysis/graph/powerdeck/pointmap_1.html"
	save(tab,outfp)


	return tab