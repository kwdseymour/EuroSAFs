#!~/anaconda3/envs/GIS/bin/python

#bsub -n 32 -R "rusage[mem=4096]" -oo $HOME/SAFlogistics/results/01_merra_wind_preprocessing 'python SAFlogistics/scripts/01_merra_wind_preprocessing/01_merra_wind_preprocessing.py -d $HOME/SAFlogistics -p32'

'''
This script opens the downloaded MERRA wind netCDF4 files for each country, preprocesses them, and saves the results to parquet files.
In the preprocessing step, the hourly wind speed is calculated and, with the windpowerlib library, a wind turbine is selected to provide the lowest LCOE for each evaluation location.
Opening of the MERRA files for some countries can require memory on the order of GB.

----- File Inputs -----
- evaluation countries: the list of countries for which the MERRA files is to be processed must be contained in a CSV file in the same directory as this script

- evaluation points: the JSON file containing the points that should be evaluated within each country should be in the following path relative to the "SAFlogistics" directory:
    "/data/Countries_WGS84/Europe_Evaluation_Points.json"

- MERRA files: these netCDF4 files must be contained in folders named according to the respsective country in the following path relative to the "SAFlogistics" directory:
    "/data/MERRA/"


----- File Outputs -----
- logs: logging files are written to the following path relative to the "scratch" directory:
    "scratch/logs/"
If this script is run on an Euler home directory, it will locate the "scratch" path via the $SCRATCH shell environment variable. Otherwise it must be located in the "SAFlogistics" directory.

- wind files: preprocessed wind power output data (parquet) files for each country are written to the following path relative to the "SAFlogistics" directory:
    "/results/01_merra_wind_preprocessing/"

'''

import os
import sys
import argparse
import numpy as np
import xarray as xr
import pandas as pd
idx = pd.IndexSlice
import re
import datetime
import windpowerlib as wpl
import logging
import time
import json

sstime = time.time() # time in seconds since January 1, 1970 (float)
sstime_string = datetime.datetime.fromtimestamp(sstime).strftime('%d%m%Y%H') # takes sstime and transfers it into date and time of local timezone  (string with day, month year and hour)
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0] # splits the pathname to get the name of the script

desc_text = 'Extract files to DataFrame, preprocess dataframe, and compute power output for each country in the "countries" list'
parser = argparse.ArgumentParser(description=desc_text) # set-up of an userfriendly command-line
parser.add_argument( # necessary input of user
    '-d','--SAF_directory',
    help='The path to the "SAFlogistics" directory',
)
parser.add_argument( # optional input of user
    '-p','--max_processes',
    help='The maximum number of subprocesses to use for executing the file operations.',
    default=12,
    type=int
)
args = parser.parse_args()

max_processes = args.max_processes
SAF_directory = args.SAF_directory

# Define path to logs, cache, and results folders. Create directories if they don't already exist
logs_path = os.path.join(SAF_directory,'logs')
if not os.path.isdir(logs_path):
    os.mkdir(logs_path)
cache_path = os.path.join(SAF_directory,'cache')
if not os.path.isdir(cache_path):
    os.mkdir(cache_path)
results_path = os.path.join(SAF_directory,'results',script_name)
if not os.path.isdir(results_path):
    os.mkdir(results_path)

# Import the Component class, which extracts plant component cost and efficiency data
sys.path.insert(1, os.path.join(SAF_directory,'scripts','03_plant_optimization'))
from plant_optimization.plant_optimizer import Component,costs_NPV

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s: %(message)s')
# File handler directed to a persistent log that is appended to.
fh1 = logging.FileHandler(os.path.join(logs_path,f'{script_name}_persistent.log'))
fh1.setLevel(logging.INFO)
fh1.setFormatter(formatter)
# File handler directed to a log that is overwritten with every script run.
fh2 = logging.FileHandler(os.path.join(logs_path,f'{script_name}.log'),mode='w')
fh2.setLevel(logging.DEBUG)
fh2.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
ch.setFormatter(formatter)

logger.addHandler(fh1)
logger.addHandler(fh2)
logger.addHandler(ch)
logger.propagate = False

# Validate countries in config file. Remove countries not found in the MERRA folder
countries_filepath = os.path.join(os.path.dirname(sys.argv[0]),f'{script_name}.csv')

try:
    countries = list(pd.read_csv(countries_filepath, index_col=0)['country'])
except FileNotFoundError:
    raise Exception('Configuration file not found. The csv file containing the list of coutries to analyze should be found in the same directory location as this script.')

removed_countries = []
for country in countries:
    if country not in str(os.listdir(os.path.join(SAF_directory,'data','MERRA'))):
        countries.remove(country)
        removed_countries.append(country)
if len(removed_countries)>0:
    logger.info('The following countries were not found in the MERRA folder: {}'.format(removed_countries))
logger.info('Country set used for analysis: {}'.format(countries))

# Read evaluation points file used to filter out points which are without country borders
try:
    with open(os.path.join(SAF_directory,'data','Countries_WGS84/Europe_Evaluation_Points.json'),'r') as fp:
        eu_eval_points = json.load(fp)
    with open(os.path.join(SAF_directory,'data','Countries_WGS84/Coast_Evaluation_Points.json'),'r') as fp:
        coast_eval_points = json.load(fp)
except FileNotFoundError:
    raise Exception('Europe_Evaluation_Points.json file not found. This file containing the MERRA points within each country\'s borders must be available at data/Countries_WGS84/.')

def extract_date(data_set):
    """
    Extracts the date from the script_name before merging the datasets. 
    (Adapted from: https://github.com/Open-Power-System-Data/weather_data/blob/2019-04-09/main.ipynb)
    """
    try:
        # The attribute name changed during the development of this script
        # from HDF5_Global.script_name to script_name. 
        if 'HDF5_GLOBAL.script_name' in data_set.attrs:
            f_name = data_set.attrs['HDF5_GLOBAL.script_name']
        elif 'script_name' in data_set.attrs:
            f_name = data_set.attrs['script_name']
        elif 'Filename' in data_set.attrs:
            f_name = data_set.attrs['Filename']
        else: 
            raise AttributeError('The attribute name has changed again!')
        
        # find a match between "." and ".nc4" that does not have "." .
        exp = r'(?<=\.)[^\.]*(?=\.nc4)'
        res = re.search(exp, f_name).group(0)
        # Extract the date. 
        y, m, d = res[0:4], res[4:6], res[6:8]
        date_str = ('%s-%s-%s' % (y, m, d))
        data_set = data_set.assign(date=date_str)
        logger.info('Running "extract_date" function.')
        return data_set

    except KeyError:
        # The last dataset is the one all the other sets will be merged into. 
        # Therefore, no date can be extracted.
        return data_set
        

def files_to_dataframe(country, coast):
    '''Extracts all files in the country folder and concatenates them by date into a DataFrame.'''
    if coast:
        file_path = os.path.join(SAF_directory, 'data', 'MERRA', country,'coast', '*.nc4')
    else:
        file_path = os.path.join(SAF_directory,'data','MERRA',country,'*.nc4')
    try:
        # with xr.open_mfdataset(file_path, concat_dim='date', combine='nested',preprocess=extract_date) as ds_wind:
        with xr.open_mfdataset(file_path, concat_dim='lat', combine='by_coords') as ds_wind:
            df_wind = ds_wind.to_dataframe()

    except xr.MergeError as e:
        logger.error('Merge error for {}:'.format(country))
        logger.error(e)

    except FileNotFoundError:
        exit()
        
    init_len = len(df_wind)
    df_wind.dropna(subset=['V10M','U10M','V50M','U50M'],how='any',inplace=True)
    if len(df_wind) < init_len:
        logger.info('{}: {} NaN rows were found and dropped.'.format(country,init_len - len(df_wind)))
    return df_wind

def preprocess_df(df_wind,country,eval_points):
    '''Resets and configures multi-index of dataframe, calculates wind velocities, and calculates Hellmann exponent.'''
    df_wind.reset_index(inplace=True)
    df_wind.loc[abs(df_wind.lon)<1e-3,'lon'] = 0 # xarray reads zeros as very small numbers, this sets them correctly
    df_wind.loc[abs(df_wind.lat)<1e-3,'lat'] = 0 # xarray reads zeros as very small numbers, this sets them correctly
    df_wind.time = pd.to_datetime(df_wind.time)
    # df_wind.drop(columns='date',inplace=True)
    df_wind.set_index(['lat','lon','time'],inplace=True)
    df_wind.sort_index(inplace=True)

    # Drop points not in "eval_points"
    df_len = len(df_wind) #Store the length for comparison
    # df_wind.drop(df_wind.loc[~df_wind.index.droplevel(2).isin([(x[1],x[0]) for x in eval_points[country]])].index,inplace=True)
    df_wind.drop(df_wind.loc[~df_wind.index.droplevel(2).isin(eval_points[country])].index,inplace=True)
    logger.info(f'Dropped {(df_len-len(df_wind))/df_len*100:.1f}% of points due to location outside {country} borders.')

    # Calculate the wind speed from the northward and eastward velocity components
    df_wind['v_10m'] = np.sqrt(df_wind['U10M']**2 + df_wind['V10M']**2) #[m/s]
    df_wind['v_50m'] = np.sqrt(df_wind['U50M']**2 + df_wind['V50M']**2) #[m/s]

    # Calculate the "Hellmann exponent", which effectively describes the instability of the atmosphere depending on surface roughness, obstacles, etc.
    # The 50 meter wind speed corresponds to that 50 m above the ground, the 10 m wind speed corresponds to that 10 m above the zero-plane displacement height (DISPH) (Mosshammer, 2016)
    df_wind['hellmann'] = (np.log(df_wind.v_50m) - np.log(df_wind.v_10m)) / (np.log(50) - np.log(10 + df_wind.DISPH)) # (Mosshammer, 2016)
    df_wind.drop(columns=['U10M','V10M','U50M','V50M'],inplace=True)

# Generate wind turbine objects
turbines = {}
turbine_classes = {}
turbine_cost_objects = {}
specs_path = os.path.join(SAF_directory,'data','plant_assumptions.xlsx')
component_specs = pd.read_excel(specs_path,sheet_name='data',index_col=0)
turbine_specs = pd.read_csv(os.path.join(os.path.dirname(wpl.__file__),'oedb','turbine_data.csv'))
all_types = list(wpl.wind_turbine.get_turbine_types('local',filter_=True,print_out=False)['turbine_type'].unique())
specific_capacity_classes = {0.2:'lo',0.3:'mid',0.47:'hi'} # 2018 JRC report wind turbing specific capacity classes used for determining costs
rep_specific_capacities = [0.2,0.3,0.47] # 2018 JRC report wind turbine class representative specific capacities
rep_hub_heights = {0.2:200, 0.3:100, 0.47:50} # 2018 JRC report wind turbine class hub heights for each of the three representative specific capacities
loading_errors = []

for model in all_types:
    '''
    Select the hub that matches most closely to the closest wind turbine category in the 2018 JRC report:
     - Turbine specific capacity of 0.2 kW/m2 (low specific capacity) and at 200 m hub height (high hub height)
     - Turbine specific capacity of 0.3 kW/m2 (medium specific capacity), at 100 m hub height (medium hub height)
     - Turbine specific capacity of 0.47 kW/m2 (high specific capacity), at 50 m hub
    '''
    try:
        specific_capacity = turbine_specs.loc[turbine_specs.turbine_type==model,'nominal_power'].item()/1e3/turbine_specs.loc[turbine_specs.turbine_type==model,'rotor_area'].item() # [kW/m2]
        hub_height_item = turbine_specs.loc[turbine_specs.turbine_type==model,'hub_height'].item().replace(' ','')
        hub_height_list = [float(x) for x in re.split(';|/|,',hub_height_item) if len(x)>0]
        rep_specific_capacity = min(rep_specific_capacities, key=lambda x:abs(x-specific_capacity)) # Selects the JRC representative specific capacity closest to that of this turbine
        rep_hub_height = rep_hub_heights[rep_specific_capacity] # Selects the representative hub height for the JRC wind turbine class closets to this turbine's specific capacity
        hub_height = min(hub_height_list, key=lambda x:abs(x-rep_hub_height)) # [m] Selects the hub height for this model that is closest to the representative hub height found above
        turbines[model] = wpl.wind_turbine.WindTurbine(turbine_type=model,hub_height=hub_height)
        turbine_classes[model] = specific_capacity_classes[rep_specific_capacity]
        if model in ['V164/8000', 'S152/6330']:
            turbine_cost_objects[model] = Component('wind',specs=component_specs,wind_class='off')
        else:
            turbine_cost_objects[model] = Component('wind', specs=component_specs, wind_class='on')
    except:
        loading_errors.append(model)

if len(loading_errors)>0:
    logger.info('There was a problem loading the following wind turbine models: {}'.format(str(loading_errors)))

#turbine_spacing_on = component_specs.loc[component_specs.index=='windon_turbine_spacing']['value'].values[0]
#turbine_spacing_off = component_specs.loc[component_specs.index=='windoff_turbine_spacing']['value'].values[0]
models=[]
specific_outputs=[]

def assign_turbine_model(df,coast,type):
    '''Calculates hypothetical power output for each turbine model in "models" and returns the model with the highest yearly output per land area usage.'''
    if coast:
        for name in list(turbines.keys()):
            if name not in['V164/8000', 'S152/6330']:
                turbines.pop(name)
    else:
        for name in list(turbines.keys()):
            if name in ['V164/8000', 'S152/6330']:
                turbines.pop(name)
    winner_model = ''
    winner_score = 0
    for name,turbine in turbines.items():
        speed_at_hub = df.v_50m*(turbine.hub_height/50)**df.hellmann
        output = sum(wpl.power_output.power_curve(speed_at_hub,turbine.power_curve.wind_speed,turbine.power_curve.value))/1e3 #kWh
        pos_output = turbine.nominal_power/1e3 * (366*24) #kWh
        share = output/pos_output
        cost_object = turbine_cost_objects[name]
        if coast:
            CAPEX = cost_object.CAPEX
            OPEX = cost_object.OPEX
            lifetime=cost_object.lifetime
        else:
            CAPEX = cost_object.CAPEX
            OPEX = cost_object.OPEX
            lifetime = cost_object.lifetime
        lcoe = costs_NPV(capex=CAPEX, opex=OPEX,discount_rate=component_specs.at['eco_discount_rate', 'value'], lifetime=lifetime,capacity=turbine.nominal_power / 1e3) / (output * cost_object.lifetime)  # EUR/kWh
        if type == 'cost':
            if lcoe > winner_score:
                winner_model = name
                winner_score = lcoe
        else:
            if share > winner_score:
                winner_model = name
                winner_score = share

    return winner_model
        
def compute_power_output(df, coast,type):
    '''Calculates the hourly power output from the wind speed data using an optimal wind turbine.

    - The optimal wind turbine is derived from the "assign_turbine_model" function.
    - The turbine type name, rated power, and rotor diameter are broadcast to new columns of the passed dataframe.
    - The power output is in MWh and is added as a column to the dataframe passed to the function
    '''
    df['kWh'] = ''
    df['turbine_type'] = ''
    df['rotor_diameter'] = ''
    df['rated_power_MW'] = ''
    df['specific_capacity_class'] = ''
    for coords in df.index.droplevel(2).unique():
        optimal_turbine = assign_turbine_model(df.loc[coords], coast,type)
        speed_at_hub = df.loc[coords].v_50m*(turbines[optimal_turbine].hub_height/50)**df.loc[coords].hellmann
        output = wpl.power_output.power_curve(speed_at_hub,turbines[optimal_turbine].power_curve.wind_speed,turbines[optimal_turbine].power_curve.value) # [Wh]
        df.loc[idx[coords],'kWh'] = list(output/1e3) # [kWh]
        df.loc[idx[coords],'turbine_type'] = optimal_turbine
        df.loc[idx[coords],'rated_power_MW'] = turbines[optimal_turbine].nominal_power/1e6 # [MW]
        df.loc[idx[coords],'rotor_diameter'] = turbines[optimal_turbine].rotor_diameter # [m]
        df.loc[idx[coords],'specific_capacity_class'] = turbine_classes[optimal_turbine]

def process_country(country, eval_points, coast, type):
    '''Extract files to DataFrame, preprocess dataframe, and compute power output for each country in the "countries" list'''
    # Create a script_name for caching the interim results: "01_merra_wind_preprocessing_<country>_<script_time>.csv"
    logger.info(f'{country} processing started...')
    cache_file_name = f'{script_name}_{country}_{sstime_string}.csv'
    # Extract files & preprocess DataFrame
    df_wind = files_to_dataframe(country, coast)
    preprocess_df(df_wind,country,eval_points)
    # Cache DatFrame
    logger.info(f'Initial caching for {country}...')
    df_wind.to_csv(os.path.join(cache_path,cache_file_name))
    logger.info(f'Initial {country} cache saved')
    # Calculate power output
    compute_power_output(df_wind, coast,type)
    # Cache & save results
    logger.info(f'Saving results for {country}...')
    df_wind.to_csv(os.path.join(cache_path,cache_file_name))
    if type == 'cost':
        if coast:
            df_wind.to_parquet(os.path.join(results_path,'optimal_cost', 'coast', f'{country}.parquet.gzip'), compression='gzip')
            df_wind.to_csv(os.path.join(results_path,'optimal_cost','coast', f'{country}.csv'))
        else:
            df_wind.to_parquet(os.path.join(results_path,'optimal_cost',f'{country}.parquet.gzip'),compression='gzip')
            df_wind.to_csv(os.path.join(results_path,'optimal_cost',f'{country}.csv'))
    else:
        if coast:
            df_wind.to_parquet(os.path.join(results_path,'optimal_out', 'coast', f'{country}.parquet.gzip'), compression='gzip')
            df_wind.to_csv(os.path.join(results_path,'optimal_out','coast', f'{country}.csv'))
        else:
            df_wind.to_parquet(os.path.join(results_path,'optimal_out',f'{country}.parquet.gzip'),compression='gzip')
            df_wind.to_csv(os.path.join(results_path,'optimal_out',f'{country}.csv'))

    logger.info(f'Results for {country} saved')

countries_eu = list(eu_eval_points.keys())

#for country in countries_eu:
    #print(country)
    #process_country(country, eu_eval_points, coast = False, type = 'cost')
    #process_country(country, eu_eval_points, coast=False, type='out')

countries_coast = list(coast_eval_points.keys())[22:]

for country in countries_coast:
    print(country)
    process_country(country, coast_eval_points, coast = True, type = 'cost')
    process_country(country, coast_eval_points, coast=True, type='out')

logger.info('Script time: {:.2f} seconds.'.format(time.time()-sstime))