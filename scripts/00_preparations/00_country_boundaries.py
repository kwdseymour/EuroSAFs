'''
This notebook processes the ESRI world countries boundaries shapefile downloaded from ArcGIS [here](https://hub.arcgis.com/datasets/a21fdb46d23e4ef896f31475217cbb08_1?geometry=96.559%2C-89.221%2C-104.535%2C86.867). 

The country shapes are pared down to include primarily only the main land masses of each country. Small islands and distant territories are removed.

The resulting shapefile is saved to:  
--- */LAV/EnergySystemsGroup/Research/Aviation/SAFlogistics/data/Countries_WGS84/Europe_WGS84.shp*

In the second part of the notebook, the evaluation grid and evalution points are determined:  
The resolution of hourly wind & PV data is limited by the MERRA wind data **(0.5 x 0.625°)**. This is the geospatial resolution used for all analysis in this notebook and beyond. Thus, each country is divided into a grid with the MERRA resolution and each cell is represented by the lat & lon coordinates from the MERRA resolution on which the cell is centered. For cells along shorelines and borders, the cells are not rectangular and the lat & lon point representing the cell may not fall within the cell boundaries (geometry). Thus, for each cell, an attempt is made to find a representative point that falls within the geometry. These points are labeled "PV_lat" and "PV_lon" because they are used in the N03_PVGIS_Download notebook to query the PVGIS API for PV data at that point (an attempt to query a point that is over the sea/ocean will return an error).

The set of gridded countries, which contain the PV_lat/PV_lon points for each cell, is saved as a shapefile to:  
--- *LAV/EnergySystemsGroup/Research/Aviation/SAFlogistics/data/Countries_WGS84/Europe_Evaluation_Grid.shp*

The set of MERRA points for each country is saved as a shapefile to:  
--- *LAV/EnergySystemsGroup/Research/Aviation/SAFlogistics/data/Countries_WGS84/Europe_Evaluation_Points.shp*  
and as a JSON file to:  
--- *LAV/EnergySystemsGroup/Research/Aviation/SAFlogistics/data/Countries_WGS84/Europe_Evaluation_Points.json*
___
'''

import geopandas as gpd
import pandas as pd
import shapely
import numpy as np
import json

# Data obtained from here: https://hub.arcgis.com/datasets/a21fdb46d23e4ef896f31475217cbb08_1?geometry=96.559%2C-89.221%2C-104.535%2C86.867
world_raw = gpd.read_file('./data/Countries_WGS84/Countries_WGS84.shp')
world = world_raw.rename(columns={'CNTRY_NAME':'name'}).drop(columns='OBJECTID')
world.name = world.name.apply(lambda x: x.replace(' ','_'))

# Import EU + EFTA country names
eu_efta = pd.read_csv('./data/EU_EFTA_Countries.csv',index_col=0)
EU_EFTA = list(eu_efta.country.unique())
europe_unfiltered = world.loc[world.name.isin(eu_efta.country.unique())].copy()
europe_unfiltered.reset_index(drop=True,inplace=True)

# Check that all europe_unfiltered countries were properly extracted
print(f'{len(eu_efta)} countries in the EU_EFTA dataset. {len(europe_unfiltered)} countries in the world DataFrame europe_unfiltered slice.')
assert len(eu_efta)==len(europe_unfiltered)

# Remove excess geometries
# The following cell removes a geometry from a country if:  
# - The land area is less than 20 km2  
#     OR  
# - The distance to the biggest geometry ("mainland") is greater than 4 times the length of the given geometry (eliminates distant territories)
removed_polygons = {}
europe = europe_unfiltered.copy()

for country in europe_unfiltered.name.unique():
    removed_polygons[country]=0
    if type( europe_unfiltered.loc[europe_unfiltered.name==country,'geometry'].item()) == shapely.geometry.multipolygon.MultiPolygon:
        multipolygon = europe_unfiltered.loc[europe_unfiltered.name==country,'geometry'].item()
        poly_df = gpd.GeoDataFrame({'geometry':list(multipolygon)})
        poly_df.crs = world.crs
        poly_df.to_crs('epsg:3035',inplace=True) # EPSG 3035 is a CRS for equal area in europe_unfiltered

        # NOTE: Using EPSG:3035 for calculating the length of geometries is imprecise but it is sufficient for the following operations
        poly_df['length_km'] = poly_df.geometry.length/1e3 # [km]
        poly_df['area_sqkm'] = poly_df.geometry.area/1e6 # [km2]
        max_length = max(poly_df.length_km)
        for idx in poly_df.index:
            poly_df[f'distance_to_main'] = poly_df.distance(poly_df.at[poly_df.area_sqkm.idxmax(),'geometry'])/1e3 # [km]
#             poly_df[f'distance_to_{idx}'] = poly_df.distance(poly_df.at[idx,'geometry'])/1e3 # [km]
        for idx in poly_df.index:
            row = poly_df.loc[idx]
            if row.area_sqkm < 20 or row.distance_to_main>row.length_km*4:
                '''Removes a geometry if:
                    - The land area is less than 20 km2
                        OR
                    - The distance to the biggest geometry ("mainland") is greater than 4 times the length of the given geometry (eliminates distant territories)
                '''
                poly_df.drop(idx,inplace=True)
                removed_polygons[country] += 1
        country_idx = europe.index[europe.name==country][0]
        geo_series = gpd.GeoSeries(index=[country_idx],data=[shapely.geometry.MultiPolygon(list(poly_df.geometry))],crs='epsg:3035')
        europe.loc[[country_idx],'geometry'] = geo_series.to_crs(crs=world.crs)
        # Verify that the new multipolygon put into the europe_unfiltered dataframe has the correct number of polygons
        assert len(europe.loc[europe.name==country,'geometry'].item()) == len(world.loc[world.name==country,'geometry'].item()) - removed_polygons[country]

europe.to_crs(crs=world.crs,inplace=True)

# Save the result to a shapefile
europe.to_file('./data/Countries_WGS84/Europe_WGS84.shp')

# PV and wind resources at every location within a country use the nearest wind/PV resource data point. These points are spaced according to the MERRA resolution. For some locations near the borders of countries, the nearest PV/wind data point will be outside the country's borders. Thus, in order to incorporate these points, a point inside the borders is found (where possible).
# Generate country grids
def generated_gridded_country(country, map_eu):
    '''Extracts the geometry of the given country and subdivides it into a grid with dimensions equivalent to the 
    MERRA data.
    Returns a GeoDataFrame'''
    node_width = 0.625
    node_height = 0.5
    if False == False:
        bounds = map_eu.loc[map_eu.name==country,'geometry'].item().bounds
        x = np.arange(np.floor(bounds[0]/node_width)*node_width,np.ceil(bounds[2]/node_width)*node_width+node_width,node_width)
        y = np.arange(np.floor(bounds[1]/node_height)*node_height,np.ceil(bounds[3]/node_height)*node_height+node_height,node_height)
        grid = []
        lats = []
        lons = []
        for i in x:
            for j in y:
                grid.append(shapely.geometry.Polygon([[i-node_width/2,j+node_height/2], # top left
                                                      [i+node_width/2,j+node_height/2], # top right
                                                      [i+node_width/2,j-node_height/2], # bottom right
                                                      [i-node_width/2,j-node_height/2]])) # bottom left
                lats.append(j)
                lons.append(i)
        grid_df = gpd.GeoDataFrame({'geometry':grid,'lat':lats,'lon':lons})
        grid_df.crs = map_eu.crs
        gridded_country = gpd.overlay(grid_df,map_eu.loc[map_eu.name==country],how='intersection')
    else:
        gridded_country = map_eu.loc[map_eu.name==country]
    return gridded_country

def get_coast_points(coast, europe_grid):
    evalp_coast = gpd.GeoDataFrame()

    coast['geometry'] = coast['geometry'].buffer(0.1)

    for row_coast in coast.iterrows():
        for row_eu in europe_grid.iterrows():
            if row_coast[1]['geometry'].intersects(row_eu[1]['geometry']) and len(evalp_coast) == 0:
                evalp_coast = evalp_coast.append(row_eu[1])
            elif row_coast[1]['geometry'].intersects(row_eu[1]['geometry']) and row_eu[0] not in evalp_coast.index:
                evalp_coast = evalp_coast.append(row_eu[1])

    return evalp_coast


coast = europe.copy()
coast.to_crs('epsg:3035',inplace=True)
coast['geometry'] = coast['geometry'].buffer(20000)

land_mass = europe.copy()
land_mass.to_crs('epsg:3035',inplace=True)
land_mass['geometry'] = land_mass['geometry'].buffer(1000)

coast = gpd.overlay(coast, land_mass, how='difference')
coast.to_crs(crs=world.crs,inplace=True)

coast = gpd.overlay(coast, world, how='difference')

# Run the function above on all countries and concatenate them to a single dataframe: europe_grid
europe_grid = gpd.GeoDataFrame({'name':[],'geometry':[],'lat':[],'lon':[]})
coast_grid = gpd.GeoDataFrame({'name':[],'geometry':[],'lat':[],'lon':[]})

for country in EU_EFTA:
    europe_grid = pd.concat([europe_grid, generated_gridded_country(country, europe)],ignore_index=True)
    if country in list(coast['name']):
        coast_grid = pd.concat([coast_grid, generated_gridded_country(country, coast)], ignore_index=True)

europe_grid.crs = europe.crs
coast_grid.crs = coast.crs

europe_grid['PV_lat'] = europe_grid.centroid.apply(lambda x: x.y) # This represents the latitude location of a representative point within each cell
europe_grid['PV_lon'] = europe_grid.centroid.apply(lambda x: x.x) # This represents the longitude location of a representative point within each cell

for idx in europe_grid.index:
    if type(europe_grid.at[idx,'geometry']) == shapely.geometry.multipolygon.MultiPolygon:
        multipolygon = europe_grid.at[idx,'geometry']
        poly_df = gpd.GeoDataFrame({'geometry':list(multipolygon)})
        poly_df.crs = world.crs
        poly_df.to_crs('epsg:3035',inplace=True) # EPSG 3035 is a CRS for equal area in europe_unfiltered

        # NOTE: Using EPSG:3035 for calculating the area of geometries is imprecise but it is sufficient for the following operations
        poly_df['area'] = poly_df.geometry.area
        poly_df.to_crs(europe_grid.crs,inplace=True)
        # Selects the centroid of the largest geometry in the multipolygon
        centroid = poly_df.sort_values('area',ascending=False).iloc[0]['geometry'].centroid
        europe_grid.loc[idx,'PV_lat'] = centroid.y
        europe_grid.loc[idx,'PV_lon'] = centroid.x

for index_1, grid_1 in coast_grid.iterrows():
    for index_2, grid_2 in coast_grid.iterrows():
        if index_1 < index_2:
            intersection = grid_1['geometry'].intersection(grid_2['geometry'])
            if intersection.area > 0:
                cut_1 = intersection.area / grid_1['geometry'].area
                cut_2 = intersection.area / grid_2['geometry'].area
                if cut_1 > cut_2:
                    new_field_2 = grid_2['geometry'] - intersection
                    if new_field_2.is_empty == False:
                        coast_grid.loc[coast_grid.index == index_2] = coast_grid.loc[coast_grid.index == index_2].set_geometry(col=[new_field_2])

                else:
                    new_field_1 = grid_1['geometry'] - intersection
                    if new_field_1.is_empty == False:
                        coast_grid.loc[coast_grid.index == index_1] = coast_grid.loc[coast_grid.index == index_1].set_geometry(col=[new_field_1])


coast_grid.to_crs('epsg:3035',inplace=True)

for index,row in coast_grid.iterrows():
    if row['geometry'].area < 20000000: # 20km²
        coast_grid = coast_grid.drop(index)

coast.to_crs(crs=world.crs, inplace=True)

# Save the points in a shapefile
coast_grid.to_file('./data/Countries_WGS84/Coast_Evaluation_Grid.shp')

europe_coast_points = get_coast_points(coast, europe_grid)
europe_coast_points.to_file('./data/Countries_WGS84/Europe_Coast_Evaluation_Grid.shp')

europe_grid.to_file('./data/Countries_WGS84/Europe_Evaluation_Grid.shp')

europe_PV_points = europe_grid.copy()
europe_PV_points['geometry'] = europe_PV_points.apply(lambda x: shapely.geometry.Point(x.PV_lon,x.PV_lat),axis=1)
europe_PV_points.to_file('./data/Countries_WGS84/Europe_Evaluation_Points.shp')

coast_points = coast_grid.copy()
coast_points['geometry'] = coast_points.apply(lambda x: shapely.geometry.Point(x.lon,x.lat),axis=1)
coast_points.to_file('./data/Countries_WGS84/Coast_Evaluation_Points.shp')

# Creates a copy of the europe_grid dataframe and reassigns the geometry column to a single MERRA point for each cell location instead of the cell shape
europe_points = europe_grid.drop(columns='geometry')
europe_points['geometry'] = europe_grid.apply(lambda x: shapely.geometry.Point(x.lon,x.lat),axis=1)

coast_points = coast_grid.drop(columns='geometry')
coast_points['geometry'] = coast_grid.apply(lambda x: shapely.geometry.Point(x.lon,x.lat),axis=1)

# Translate the points into a dictionary to save as a JSON
europe_points_dict = {}
coast_points_dict = {}

for country in europe_points['name'].unique().tolist():
    europe_points_dict[country] = []

for country in coast_points['name'].unique().tolist():
    coast_points_dict[country] = []

for idx in europe_points.index:
    europe_points_dict[europe_points.at[idx,'name']].append((europe_points.at[idx,'geometry'].y,europe_points.at[idx,'geometry'].x))
for idx in coast_points.index:
    coast_points_dict[coast_points.at[idx, 'name']].append((coast_points.at[idx, 'geometry'].y, coast_points.at[idx, 'geometry'].x))

# Save the list of points to a JSON file
# The points are saved as tuples: (lat,lon)
with open('./data/Countries_WGS84/Europe_Evaluation_Points.json', 'w') as fp:
    json.dump(europe_points_dict, fp)

with open('./data/Countries_WGS84/Coast_Evaluation_Points.json', 'w') as fp:
    json.dump(coast_points_dict, fp)