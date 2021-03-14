# python scripts/data_preparation/PVGIS_download.py

'''
This notebook utilizes the non-interactive (API) [European Commission's PVGIS tool](https://ec.europa.eu/jrc/en/pvgis) to download hourly PV performance data for 2016. As the tool allows sampling of any coordinate, the resolution of hourly wind & PV performance data is restricted by the resolution of the MERRA wind data, which has a resolution of  0.5 x 0.625°. Thus, the points of interest are first obtained from the MERRA wind datasets: for each country, the downloaded MERRA coordinate points are extracted.

Then, PVGIS API is utilized to extract the following features for each coordinate point (see: [API reference](https://ec.europa.eu/jrc/en/PVGIS/docs/noninteractive) and [output documentation](https://ec.europa.eu/jrc/en/PVGIS/tools/hourly-radiation)):
- Wh [Wh] - hourly power output of a PV installation per kW of installed capacity of optimally aligned single horizontal axis aligned north-south PV panels ([datasources and calculation methods](https://ec.europa.eu/jrc/en/PVGIS/docs/methods))
- G(i) [W/m2] - Global in-plane irradiance
- H_sun [º] - Sun height (elevation)
- T2m [°C] - Air temperature
- WS10m [m/s] - Wind speed at 10m

System assumptions:
- PV panels are optimally aligned single horizontal axis aligned north-south panels
- Sum of system losses = 14% (this should be reviewed?)

The databases used to calculate radiation depends on the location being queried: PVGIS-SARAH for most of Europe and PVGIS-ERA5 for European latitudes above 60 N. 
See more the "raddatabase" parameter description in the [API reference](https://ec.europa.eu/jrc/en/PVGIS/docs/noninteractive) and chapter 3 of the [PVGIS users manual](https://ec.europa.eu/jrc/en/PVGIS/docs/usermanual#fig:default_db).

The results are saved as parquet files for each country in:  
/data/PVGIS/
'''

import requests
import pandas as pd
import matplotlib.pyplot as plt
import time
import geopandas as gpd
import aiohttp
import asyncio
import logging
import os
import sys

# Get EuroSAFs parent directory 
SAF_directory = os.path.dirname(__file__)
for i in range(2):
    SAF_directory = os.path.dirname(SAF_directory)

# Get the path to the scratch directory. This is where logging files will be saved
if 'cluster/home' in os.getcwd():
    # Uses the $SCRATCH environment variable to locate the scratch file if this module is run within Euler
    scratch_path = os.environ['SCRATCH']
else:
    scratch_path = os.path.join(SAF_directory,'scratch')
    
# Add a logger
sys.path.append(os.path.join(SAF_directory,'scripts/optimization'))
from plant_optimization.utilities import create_logger
logger = create_logger(scratch_path,__name__,__file__)

europe_pv_points = gpd.read_file(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_PV_Evaluation_Points.shp'))

EU_EFTA = list(pd.read_csv(os.path.join(SAF_directory,'data/EU_EFTA_Countries.csv'),index_col=0).country)

# API Requests

results_dict = {} # Holds the DataFrames of the extracted data
error_points = [] # These points encountered a problem on their third API request attempt
sea_points = [] # These points were found to be over the sea

async def download_data(point,PV_eval_loc,sema,session,year=2016):
    '''
    Query the PVGIS API for the given "point" at the given "PV_eval_loc".
    The output is saved to the results_dict dictionary
    '''
    for i in range(2):
        # Try the query two times to circumvent bad responses that might be returned for random reasons
        parameters = {
                        'startyear':year,
                        'endyear':year,
                        'pvcalculation':1, 
                        'peakpower':1, # Nominal power of the PV system, in kW.
                        'loss':14,
                        'trackingtype':1,
                        'optimalinclination':1,
                        'outputformat':'json'
                        }
        parameters['lat']=f'{PV_eval_loc[0]:.3f}'; parameters['lon']=f'{PV_eval_loc[1]:.3f}'
        try:
            async with sema:
                async with session.get('https://re.jrc.ec.europa.eu/api/seriescalc',params=parameters) as resp:
                    status = resp.status
                    response = await resp.json()
                    if status == 200: # 200 means the request returned a response correctly. all others indicate an error
                        df = pd.DataFrame(pd.json_normalize(response['outputs']['hourly']))
                        df['lat'] = point[0]
                        df['lon'] = point[1]
                        df['year'] = year
                        results_dict[point] = df
                        break
                    else: 
                        if 'sea' in response['message']:
                            # If the point was found to be over the sea, the message will indicate that
                            sea_points.append(point)
                            break
                        else:
                            raise Exception(response['message'])            
        except Exception as e:
            eval_lat = parameters['lat']
            eval_lon = parameters['lon']
            logger.info(f'Error with {point} (eval: {eval_lat},{eval_lon}). {e}. Retrying query for year {year}...')
            if i==0:
                logger.info(f'Error with {point} (eval: {eval_lat},{eval_lon}). {e}. Retrying...')
            else:
                logger.info(f'Error with {point} (eval: {eval_lat},{eval_lon}) on second query. {e}. Point not saved.')

async def main(country_points,year=2016):
    tasks = []
    sema = asyncio.Semaphore(10) # Limits the number of asynchronous calls to the API possible. Try lowering this number if you encounter connection issues
    async with aiohttp.ClientSession() as session:
        for idx in country_points.index:
            
            # Get the nominal point
            point = idx

            # Get the evaluation point
            PV_eval_loc = (country_points.loc[idx,'pv_lat'],country_points.loc[idx,'pv_lon'])

            task = asyncio.ensure_future(download_data(point=point,PV_eval_loc=PV_eval_loc,sema=sema,session=session,year=year))
            tasks.append(task)
        responses = asyncio.gather(*tasks)
        await responses 

# Notice: Running the following  will query the API and save the results to files

for i,country in enumerate(EU_EFTA):
    # --- Setup ---
    # Skip countries thata already had results in the data folder
    if os.path.isfile(os.path.join(SAF_directory,'data/PVGIS/'+country+'_PV.parquet.gzip')):
        logger.info(f'{country} file already found.')
        continue

    logger.info(f'Starting {country}...')

    # Get points for the current country
    country_points = europe_pv_points.loc[(europe_pv_points.country==country)&(~europe_pv_points.sea_node)].set_index(['grid_lat','grid_lon'])
    logger.info(f'{len(country_points)} points will be queried for {country}')

    stime = time.time()

    # Clear previous results
    results_dict.clear()
    sea_points.clear()
    error_points.clear()
    if len(results_dict)+ len(sea_points) + len(error_points) > 0:
        logger.error(f'Lists or dictionaries not cleared properly during {country} evaluation.')
        break
    
    # --- Run queries ---
    asyncio.run(main(country_points))

    # --- Log results ---
    logger.info(f'Queries finished after {time.time()-stime:.1f} seconds')
    logger.info(f'{len(sea_points)} sea points for {country}')
    logger.info(f'{len(error_points)} error points for {country}')
    logger.info(f'{len(results_dict)} points were succesfully queried for {country} (out of {len(country_points)}).')

    # --- Concatenate the data returned for each point into a single dataframe (results_df) ---
    if len(results_dict)>0:
        results_df = pd.concat(results_dict.values())
        # Rename "P" (power, [Watt per kW installed]) column to 'Wh' (energy produced during the given hour, [Watt-hour per kW installed])
        results_df.rename(columns={'P':'Wh'},inplace=True)
        results_df['time'] = pd.to_datetime(results_df['time'],format='%Y%m%d:%H%M')
        results_df.set_index(['lat','lon','time'],inplace=True)
        results_df.sort_index(inplace=True)
        
        # --- Check for errors ---
        results_points = list(results_df.index.droplevel(2).unique())
        wrong_points = [x for x in results_points if x not in country_points.index]
        if len(wrong_points) > 0:
            logger.error(f'{len(wrong_points)} wrong points were saved for {country}.')
            break
    else: # For some countries, it is possible that the queries return no successful responses. For these, we create an empty dataframe
        results_df = pd.DataFrame({'lat':[],'lon':[],'time':[],'Wh':[],'G(i)':[],'H_sun':[],'T2m':[],'WS10m':[],'Int':[],'year':[]})
        results_df.set_index(['lat','lon','time'],inplace=True)

    # --- Save the results to a parquet file ---
    results_df.to_parquet(os.path.join(SAF_directory,'data/PVGIS',country+'_PV.parquet.gzip'),compression='gzip')
    logger.info(f'{country} file saved.')
    logger.info(f'{country} finished after {time.time()-stime:.1f} seconds')
logger.info('All queries finished')
print('All queries finished')