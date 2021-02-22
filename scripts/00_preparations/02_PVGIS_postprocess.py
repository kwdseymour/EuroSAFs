# python scripts/00_preparations/02_PVGIS_postprocess.py

'''
This script identifies location for which the PV data appears to be erroneous. They are identified by first calculating the total yearly energy production of all locations, and then looking for outliers.
A location is considered an outlier if the yearly energy production is greater than 4 standard deviations away from the median.
For such outlier points, the nearest valid point with the same latitude is chosen to impute data. It simply replaces the erroneous data.

The results are saved at: data/PVGIS/
'''

import pandas as pd
import geopandas as gpd
import os
import sys
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

europe_grid = gpd.read_file(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_Evaluation_Grid.shp'))
# This should be tuned by trial and error
z_score_filter = 4

for country in europe_grid.country.unique():
    if os.path.isfile(os.path.join(SAF_directory,'data/PVGIS/'+country+'_PV.parquet.gzip')):
        logger.info(f'{country} file already found.')
        continue
    logger.info(f'Starting processing of {country}.')
    warnings.simplefilter("ignore", UserWarning)
    # Load data from PVGIS API
    pv_data = pd.read_parquet(os.path.join(SAF_directory,f'data/PVGIS/API_raw/{country}_PV.parquet.gzip'))
    warnings.simplefilter("default", UserWarning)
    
    # Sum data over the whole year for each point and merge with the Europe grid geodataframe to assign data to evaluation points
    pv_grp = pv_data.groupby(level=[0,1]).sum().reset_index()
    pv_gdf = europe_grid.loc[(europe_grid.country==country)&(~europe_grid.sea_node)].copy()
    pv_gdf = pv_gdf.merge(pv_grp,left_on=['grid_lat','grid_lon'],right_on=['lat','lon'],how='left')

    # Identify outliers by finding those with yearly electricity production [Wh] with a z-score greater than 4 (the energy production is more than 4 times the standard deviation from the median energy production)
    outliers = (abs(pv_gdf.Wh-pv_gdf.Wh.median())>pv_gdf.Wh.std()*z_score_filter)|pv_gdf.Wh.isna()
    outliers_df = pv_gdf.loc[outliers].copy()
    logger.info(f'{len(outliers_df)} outliers found in {country}')
    if len(outliers_df) == 0:
        # Simply save the data if no outliers are found
        pv_data.to_parquet(os.path.join(SAF_directory,f'data/PVGIS/{country}_PV.parquet.gzip'),compression='gzip')
    else:
        pv_data.reset_index(inplace=True)
        nonoutliers_df = pv_gdf.loc[~outliers].copy()
        for idx,row in outliers_df.iterrows():
            # Get the latitudes of all locations that are not outliers and identify the one closest to the given point
            valid_lats = nonoutliers_df.lat.unique()
            closest_lat = min(valid_lats, key=lambda x: abs(x-row.grid_lat))
            
            # Get all longitudes of all locations that are not outliers and have the latitude found directly above. Then find the closest one.
            same_lons = nonoutliers_df.loc[nonoutliers_df.grid_lat==closest_lat,'lon'].unique()
            closest_lon = min(same_lons, key=lambda x: abs(x-row.grid_lon))

            # Make a copy of the data at the location with the lat & lon found directly above
            data_copy = pv_data.loc[(pv_data.lat==closest_lat)&(pv_data.lon==closest_lon)].copy()
            data_copy['lat'] = row.grid_lat
            data_copy['lon'] = row.grid_lon

            # Remove existing data for the given point and append the copied data
            existing_data_idxs = (pv_data.lat==row.grid_lat)&(pv_data.lon==row.grid_lon)
            pv_data = pv_data.loc[~existing_data_idxs]
            pv_data = pv_data.append(data_copy)
        pv_data.set_index(['lat','lon','time']).to_parquet(os.path.join(SAF_directory,f'data/PVGIS/{country}_PV.parquet.gzip'),compression='gzip')
    logger.info(f'Processing of {country} finished.')