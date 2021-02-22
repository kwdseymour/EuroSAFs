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

# python scripts/00_preparations/00_country_boundaries.py

import geopandas as gpd
import pandas as pd
import shapely
import numpy as np
import json
import logging
import os
import sys 
import pyproj
import warnings

# Get EuroSAFs parent directory 
SAF_directory = os.path.dirname(__file__)
for i in range(2):
    SAF_directory = os.path.dirname(SAF_directory)
# Get scratch path
if 'cluster/home' in os.getcwd():
    # Uses the $SCRATCH environment variable to locate the scratch file if this module is run within Euler
    scratch_path = os.environ['SCRATCH']
else:
    scratch_path = os.path.join(SAF_directory,'scratch')

# Add a logger
sys.path.append(os.path.join(SAF_directory,'scripts/03_plant_optimization'))
from plant_optimization.utilities import create_logger
logger = create_logger(scratch_path,__name__,__file__)

# Data obtained from here: https://hub.arcgis.com/datasets/a21fdb46d23e4ef896f31475217cbb08_1?geometry=96.559%2C-89.221%2C-104.535%2C86.867
world_raw = gpd.read_file(os.path.join(SAF_directory,'data/Countries_WGS84/Countries_WGS84.shp'))
world = world_raw.rename(columns={'CNTRY_NAME':'country'}).drop(columns='OBJECTID')
world.country = world.country.apply(lambda x: x.replace(' ','_'))
world_crs = world.crs

# Import EU + EFTA country names
eu_efta = pd.read_csv(os.path.join(SAF_directory,'data/EU_EFTA_Countries.csv'),index_col=0)
EU_EFTA = list(eu_efta.country.unique())
europe_unfiltered = world.loc[world.country.isin(eu_efta.country.unique())].copy()
europe_unfiltered.reset_index(drop=True,inplace=True)

# Check that all europe_unfiltered countries were properly extracted
logger.info(f'{len(eu_efta)} countries in the EU_EFTA dataset. {len(europe_unfiltered)} countries in the world DataFrame europe_unfiltered slice.')
assert len(eu_efta)==len(europe_unfiltered)

# Set the minimum distance from shore in which offshore wind parks can be built
minimum_offshore_dist = 5 #[km]

# Remove excess geometries
# The following removes a geometry from a country if:  
# - The land area is less than 20 km2  
#     OR  
# - The distance to the biggest geometry ("mainland") is greater than 4 times the length of the given geometry (eliminates distant territories)
removed_polygons = {}
europe = europe_unfiltered.copy()

for country in europe_unfiltered.country.unique():
    removed_polygons[country]=0
    if type( europe_unfiltered.loc[europe_unfiltered.country==country,'geometry'].item()) == shapely.geometry.multipolygon.MultiPolygon:
        multipolygon = europe_unfiltered.loc[europe_unfiltered.country==country,'geometry'].item()
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
        country_idx = europe.index[europe.country==country][0]
        geo_series = gpd.GeoSeries(index=[country_idx],data=[shapely.geometry.MultiPolygon(list(poly_df.geometry))],crs='epsg:3035')
        europe.loc[[country_idx],'geometry'] = geo_series.to_crs(crs=world.crs)
        # Verify that the new multipolygon put into the europe_unfiltered dataframe has the correct number of polygons
        assert len(europe.loc[europe.country==country,'geometry'].item()) == len(world.loc[world.country==country,'geometry'].item()) - removed_polygons[country]
logger.info(f'Removed polygons: {removed_polygons}')

europe.to_crs(crs=world.crs,inplace=True)

# Create a buffered land mass around Europe
warnings.simplefilter("ignore", UserWarning)
buffered_europe_land_masses = world.unary_union.intersection(europe.buffer(10).unary_union.envelope)
warnings.simplefilter("default", UserWarning)

# Save the result to a shapefile
europe.to_file(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_WGS84.shp'))

# PV and wind resources at every location within a country use the nearest wind/PV resource data point. These points are spaced according to the MERRA resolution. 
# For some locations near the borders of countries, the nearest PV/wind data point will be outside the country's borders. Thus, in order to incorporate these points, a point inside the borders is found (where possible).
# Generate country grids
def generated_gridded_country(country, map_eu):
    '''Extracts the geometry of the given country and subdivides it into a grid with dimensions equivalent to the 
    MERRA data.
    Returns a GeoDataFrame'''
    node_width = 0.625
    node_height = 0.5

    # Buffer the country geometry by 60 km and find the bounds of the resulting geometry
    country_geom = map_eu.loc[map_eu.country==country,'geometry'].item()
    cent_lon = country_geom.centroid.x
    cent_lat = country_geom.centroid.y
    if country=='Spain':
        cent_lat += 1
    # The following line is adapted from here: https://automating-gis-processes.github.io/site/2018/notebooks/L2/projections.html
    aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=cent_lat, lon_0=cent_lon).srs # Create a CRS from the Azimuthal Equidistant project centered on the country's centroid
    country_geom_aeqd = map_eu.loc[map_eu.country==country].copy().to_crs(crs=aeqd)
    buffered_country = country_geom_aeqd.buffer(100000).to_crs(map_eu.crs) # Create a geometry that is buffered 100km around the original country geometry

    bounds = buffered_country.bounds.iloc[0]
    x = np.arange(np.floor(bounds[0]/node_width)*node_width,np.ceil(bounds[2]/node_width)*node_width+node_width,node_width)
    y = np.arange(np.floor(bounds[1]/node_height)*node_height,np.ceil(bounds[3]/node_height)*node_height+node_height,node_height)
    # generate list of polygons that make up a grid with extents defined by the bounds found previously and node heights/widths defined above
    grid = []
    lats = []
    lons = []
    in_country = []
    pt_in_sea = []
    for i in x:
        for j in y:
            grid.append(shapely.geometry.Polygon([[i-node_width/2,j+node_height/2], # top left
                                                  [i+node_width/2,j+node_height/2], # top right
                                                  [i+node_width/2,j-node_height/2], # bottom right
                                                  [i-node_width/2,j-node_height/2]])) # bottom left
            lats.append(j)
            lons.append(i)
            # Check if the point is within the country geometry
            in_country.append(map_eu.loc[map_eu.country==country,'geometry'].item().contains(shapely.geometry.Point(i,j)))
            pt_in_sea.append(not buffered_europe_land_masses.contains(shapely.geometry.Point(i,j)))
    grid_df = gpd.GeoDataFrame({'geometry':grid,'grid_lat':lats,'grid_lon':lons,'in_country':in_country,'pt_in_sea':pt_in_sea},crs=map_eu.crs)
    # Find the intersection of the grid with the country geometry
    land_grid = gpd.overlay(grid_df,map_eu.loc[map_eu.country==country],how='intersection')
    land_grid.crs = map_eu.crs
    land_grid['coast_pt']=False
    land_grid['sea_node']=False
    # Find the intersection of the grid with the country geometry
    sea_grid = gpd.overlay(grid_df,gpd.GeoDataFrame({'geometry':buffered_country},crs=map_eu.crs),how='intersection')
    sea_grid.crs = map_eu.crs
    # Subtract all land masses from the buffered gridded country to find the gridded coasts around the country
    # First, create a world gdf and buffer the geometries by the previously defined minimum_offshore_dist. This will be used to subtract out all land masses and shorelines within a specific distance of the coasts
    # Create bounding box gdf to limit extents of the world buffer. This avoids a problem with reprojection
    bbox = gpd.GeoDataFrame({'geometry':[shapely.geometry.Polygon([[-2e6,-2e6],[2e6,-2e6],[2e6,2e6],[-2e6,2e6]])]},crs=aeqd)
    buffered_world = gpd.overlay(world.to_crs(aeqd),bbox,how='intersection')
    buffered_world['geometry'] = buffered_world.buffer(minimum_offshore_dist*1e3)
    sea_grid = gpd.overlay(sea_grid,buffered_world.to_crs(map_eu.crs),how='difference')
    sea_grid['coast_pt'] = sea_grid.to_crs(aeqd).distance(country_geom_aeqd.geometry.item())<=minimum_offshore_dist*1e3
    sea_grid.crs = map_eu.crs
    sea_grid['country'] = country
    sea_grid['sea_node']=True
    # Combine to single df
    gridded_country = pd.concat([land_grid,sea_grid]).reset_index(drop=True)
    return gridded_country

def assign_centroids(country_df_raw):
    '''This function finds the centroid for each geometry in the country_df.
    Because some geometries are multipolygons, it first identifies the largest polygon for each entry and then finds the centroid of that geometry. 
    '''
    country_df = country_df_raw.copy()
    for idx,row in country_df.iterrows():
        if type(row.geometry) == shapely.geometry.multipolygon.MultiPolygon:
            # Break up the multipolygon and make each geometry an entry in a new dataframe, "poly_df"
            multipolygon = row.geometry
            poly_df = gpd.GeoDataFrame({'geometry':list(multipolygon)},crs = world.crs)
            # Convert CRS and find area of each geometry
            poly_df.to_crs('epsg:3035',inplace=True) # EPSG 3035 is a CRS for equal area in europe_unfiltered
            # NOTE: Using EPSG:3035 for calculating the area of geometries is imprecise but it is sufficient for the following operations
            poly_df['area'] = poly_df.geometry.area
            poly_df.to_crs(world.crs,inplace=True)
            # Selects the centroid of the largest geometry in the multipolygon
            centroid = poly_df.sort_values('area',ascending=False).iloc[0]['geometry'].centroid
            country_df.loc[idx,'cent_lat'] = centroid.y
            country_df.loc[idx,'cent_lon'] = centroid.x
        else:
            country_df.loc[idx,'cent_lat'] = row.geometry.centroid.y
            country_df.loc[idx,'cent_lon'] = row.geometry.centroid.x
    return country_df

def assign_merra_points(country_df_raw):
    '''Assigns the nearest valid MERRA point for locations on the shore where the nominal MERRA point is wrong.
    For offshore nodes in which the nominal MERRA point resides on land, the next closest offshore MERRA point it found.
    For onshore nodes in which the nominal MERRA point resides offshore, the next closest onshore MERRA point is found.
    The results are saved to new columns: "merra_lat" and "merra_lon".
    '''
    country_df = country_df_raw.copy()
    midlat = np.mean([country_df.grid_lat.min(),country_df.grid_lat.max()]) # Find the midpoint of the country's latitudes
    midlon = np.mean([country_df.grid_lon.min(),country_df.grid_lon.max()]) # Find the midpoint of the country's longitudes
    # The following line is adapted from here: https://automating-gis-processes.github.io/site/2018/notebooks/L2/projections.html
    aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=midlat, lon_0=midlon).srs # Create a CRS from the Azimuthal Equidistant project centered on the country's midpoint
   # Duplicate the dataframe and convert each grid cell's centroid to the geometry column
    centroids = country_df.copy()
    centroids['geometry'] = centroids.apply(lambda x: shapely.geometry.Point(x.cent_lon,x.cent_lat),axis=1)
    centroids.to_crs(aeqd,inplace=True)
    # Duplicate the dataframe and convert each grid cell's nominal position to the geometry column
    grid_points = country_df.copy()
    grid_points['geometry'] = grid_points.apply(lambda x: shapely.geometry.Point(x.grid_lon,x.grid_lat),axis=1)
    grid_points.to_crs(aeqd,inplace=True)

    # For nodes that are offshore (onshore) but have nominal MERRA points that are onshore (offshore), find the index of nearest offshore (onshore) point
    merra_pts_idxs = centroids.apply(lambda x: grid_points.loc[grid_points.pt_in_sea==x.sea_node].distance(x.geometry).idxmin(),axis=1) 
    country_df['merra_lat'] = merra_pts_idxs
    # Look up and assign the latitude coordinate of the nearest merra point that lies within the country's borders
    country_df['merra_lat'] = country_df.apply(lambda x: country_df.at[x.merra_lat,'grid_lat'],axis=1) 
    country_df['merra_lon'] = merra_pts_idxs
    # Look up and assign the longitude coordinate of the nearest merra point that lies within the country's borders
    country_df['merra_lon'] = country_df.apply(lambda x: country_df.at[x.merra_lon,'grid_lon'],axis=1)
    return country_df

def get_offshore_distance(country_df_raw,europe_df,country):
    ''' Determines the distance (in km) from the centroid of each offshore node to the closest shoreline and saves the result in a new column, "centroid_offshore_dist_km". 
    Note: country_df and country_geometry must have the same CRS!!!'''
    country_df = country_df_raw.copy()
    # Create an equal distance CRS centered on the centroid for the given country
    country_geom = europe_df.loc[europe_df.country==country,'geometry'].item()
    cent_lon = country_geom.centroid.x
    cent_lat = country_geom.centroid.y
    # The following line is adapted from here: https://automating-gis-processes.github.io/site/2018/notebooks/L2/projections.html
    aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=cent_lat, lon_0=cent_lon).srs # Create a CRS from the Azimuthal Equidistant project centered on the country's centroid
    
    # Get the geometry of the country in the newly created CRS
    europe_aeqd = europe.to_crs(aeqd)
    country_geom_aeqd = europe_aeqd.loc[europe_aeqd.country==country,'geometry'].item()
    
    # Convert the nominal grid coordintates to the new CRS in a new geodataframe, "points_df_aeqd"
    points_df = country_df.copy()
    points_df['geometry'] = points_df.apply(lambda x: shapely.geometry.Point(x.cent_lon,x.cent_lat),axis=1)
    points_df_aeqd = points_df.to_crs(aeqd)
    
    # Calculate the distance (in km) from each point to the country's geometry
    country_df['shore_dist'] = points_df_aeqd.distance(country_geom_aeqd)/1e3
    # Make sure offshore nodes have a distance of at least 5 km
    country_df.loc[(country_df.sea_node)&(country_df.shore_dist<minimum_offshore_dist),'shore_dist'] = minimum_offshore_dist
    
    return country_df

# Run the functions above on all countries and concatenate them to a single dataframe: europe_grid
# The result may contain offshore nodes that should belong to other countries so it is labeled "unclean"
europe_grid_unclean = gpd.GeoDataFrame()
for country in EU_EFTA:
    logger.info(f'Starting processing for {country}...')
    country_df = generated_gridded_country(country,europe)
    country_df = assign_centroids(country_df)
    country_df = assign_merra_points(country_df)
    country_df = get_offshore_distance(country_df,europe,country)
    europe_grid_unclean = pd.concat([europe_grid_unclean, country_df],ignore_index=True)
    logger.info(f'{country} finished')

# Clean the above result by removing nodes that belong to other countries' coasts
logger.info(f'Cleaning results.')
europe_grid = europe_grid_unclean.copy()
for country in EU_EFTA:
    # Get all offshore points for the country that are not directly along the coast
    non_coast_offshore = europe_grid_unclean.loc[((europe_grid_unclean.country==country)&(europe_grid_unclean.sea_node)&(~europe_grid_unclean.coast_pt))]
    # Get all offshore points of all other countries that are along their coasts
    all_other_coasts = europe_grid_unclean.loc[(europe_grid_unclean.country!=country)&(europe_grid_unclean.coast_pt)]
    # Find which points in non_coast_offshore are also in all_other_coasts
    belongs_to_other = non_coast_offshore.apply(lambda x: len(all_other_coasts[(all_other_coasts.grid_lat==x.grid_lat)&(all_other_coasts.grid_lon==x.grid_lon)])>0,axis=1)
    # Drop the points for the given country that belong to any other country's coast
    if len(belongs_to_other)>0:
        europe_grid.drop(belongs_to_other.loc[belongs_to_other].index,inplace=True)
        logger.info(f'Dropped {belongs_to_other.sum()} nodes in {country} because they belong to the coastline of another country.')

logger.info(f'Results cleaned.')

europe_grid['pv_lat'] = europe_grid['cent_lat']
europe_grid['pv_lon'] = europe_grid['cent_lon']
europe_grid.loc[europe_grid.sea_node,['pv_lat','pv_lon']] = np.nan
europe_grid.to_crs(world.crs,inplace=True)
europe_grid.reset_index(drop=True,inplace=True)

# Save the points to shapefiles
logger.info(f'Saving results.')
europe_grid.to_file(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_Evaluation_Grid.shp'))

europe_grid.loc[~europe_grid.sea_node].reset_index(drop=True).to_file(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Onshore_Evaluation_Grid.shp'))
europe_grid.loc[europe_grid.sea_node].reset_index(drop=True).to_file(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Offshore_Evaluation_Grid.shp'))

europe_points = europe_grid.copy()
europe_points['geometry'] = europe_points.apply(lambda x: shapely.geometry.Point(x.grid_lon,x.grid_lat),axis=1)
europe_points.to_file(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_Evaluation_Points.shp'))
europe_points.to_csv(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_Evaluation_Points.csv'))

europe_PV_points = europe_grid.copy()
europe_PV_points['geometry'] = europe_PV_points.apply(lambda x: shapely.geometry.Point(x.pv_lon,x.pv_lat),axis=1)
europe_PV_points.to_file(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_PV_Evaluation_Points.shp'))

europe_merra_points = europe_grid.copy()
europe_merra_points['geometry'] = europe_merra_points.apply(lambda x: shapely.geometry.Point(x.merra_lon,x.merra_lat),axis=1)
europe_merra_points.to_file(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_MERRA_Evaluation_Points.shp'))

# Translate the evaluation points into a dictionary to save as a JSON
europe_points_dict = {}
pv_points_dict = {}
merra_points_dict = {}

for country in europe_points['country'].unique():
    df = europe_points.loc[europe_points.country==country].sort_values(['grid_lat','grid_lat']).drop_duplicates(subset=['geometry'])
    europe_points_dict[country] = [(i.geometry.y,i.geometry.x) for _,i in df.iterrows()]

for country in europe_PV_points['country'].unique():
    df = europe_PV_points.loc[europe_PV_points.country==country]
    df = df.loc[~df.sea_node].sort_values(['grid_lat','grid_lat']).drop_duplicates(subset=['geometry'])
    pv_points_dict[country] = [(i.geometry.y,i.geometry.x) for _,i in df.iterrows()]

for country in europe_merra_points['country'].unique():
    df = europe_merra_points.loc[europe_merra_points.country==country].sort_values(['grid_lat','grid_lat']).drop_duplicates(subset=['geometry'])
    merra_points_dict[country] = [(i.geometry.y,i.geometry.x) for _,i in df.iterrows()]


# Save the list of points to a JSON file
# The points are saved as tuples: (lat,lon)
with open(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_Evaluation_Points.json'), 'w') as fp:
    json.dump(europe_points_dict, fp)

with open(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_PV_Evaluation_Points.json'), 'w') as fp:
    json.dump(pv_points_dict, fp)
    
with open(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_MERRA_Evaluation_Points.json'), 'w') as fp:
    json.dump(merra_points_dict, fp)

logger.info('Script finished.')