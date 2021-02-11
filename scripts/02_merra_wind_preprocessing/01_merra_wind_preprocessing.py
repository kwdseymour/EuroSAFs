#bsub -n 32 -R "rusage[mem=4096]" -oo $HOME/SAFlogistics/results/01_merra_wind_preprocessing 'python SAFlogistics/scripts/01_merra_wind_preprocessing/01_merra_wind_preprocessing.py -d $HOME/SAFlogistics -p32'

# python scripts/02_merra_wind_preprocessing/01_merra_wind_preprocessing.py -d .

'''
This script opens the downloaded MERRA wind netCDF4 files for each country, preprocesses them, and saves the results to parquet files.
In the preprocessing step, the hourly wind speed is calculated and, with the windpowerlib library, a wind turbine is selected to provide the lowest LCOE for each evaluation location.
Opening of the MERRA files for some countries can require memory on the order of GB.

----- File Inputs -----
- evaluation countries: the list of countries for which the MERRA files is to be processed must be contained in a CSV file in the same directory as this script

- evaluation points: the JSON file containing the points that should be evaluated within each country should be in the following path relative to the "SAFlogistics" directory:
    "/data/Countries_WGS84/processed/Europe_Evaluation_Points.json"

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
import multiprocessing as mp
import geopandas as gpd

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
    '-m','--optimization_metric',
    help='The metric by which the turbine selection should be optimized. Must be one of the following: "lcoe", "flh", or "density".',
    default="flh",
    type=str
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
optimization_metric = args.optimization_metric

# Define path to logs, cache, and results folders. Create directories if they don't already exist
logs_path = os.path.join(SAF_directory,'scratch','logs')
if not os.path.isdir(logs_path):
    os.mkdir(logs_path)
cache_path = os.path.join(SAF_directory,'scratch','cache')
if not os.path.isdir(cache_path):
    os.mkdir(cache_path)
results_path = os.path.join(SAF_directory,'results',script_name)
if not os.path.isdir(results_path):
    os.mkdir(results_path)
if 'cluster/home' in os.getcwd():
    # Uses the $SCRATCH environment variable to locate the scratch file if this module is run within Euler
    scratch_path = os.environ['SCRATCH']
else:
    scratch_path = os.path.join(SAF_directory,'scratch')

# Import the Component class, which extracts plant component cost and efficiency data
sys.path.insert(1, os.path.join(SAF_directory,'scripts','03_plant_optimization'))
from plant_optimization.plant_optimizer import Component,costs_NPV

# Add a logger
from plant_optimization.utilities import create_logger
logger = create_logger(scratch_path,__name__,__file__)

class Turbine(wpl.wind_turbine.WindTurbine):
    '''Child class of the windpowerlib.wind_turbine.WindTurbine class.
    turbine_type: a turbine model available in the windpowerlib library
    onshore: TRUE or False, indicates whether the wind turbine is an onshore of offshore variant
    classifications: a DataFrame indicating whether each model has on/offshore variants as well as the IEC class of the wind turbine (from classifications.csv)
    component_specs: a Dataframe holding the cost specifications of wind turbine classes (from plant_assumptions.xlsx)
    turbine_specs: the oedb DataFrame from the windpowerlib package holding information about the possible hub heights of wind turbines among others
    hub_height: can be specified (in meters) but may be overridden by the "assign_site" function
    wpl_kwargs: are any arguments accepted by the windpowerlib.wind_turbine.WindTurbine class

    Initially selects the hub height that matches most closely to the closest wind turbine category in the 2018 JRC report:
     - Turbine specific capacity of 0.2 kW/m2 (low specific capacity) and at 200 m hub height (high hub height)
     - Turbine specific capacity of 0.3 kW/m2 (medium specific capacity), at 100 m hub height (medium hub height)
     - Turbine specific capacity of 0.47 kW/m2 (high specific capacity), at 50 m hub
    '''
    specific_capacity_classes = {0.2:'lo',0.3:'mid',0.47:'hi'} # 2018 JRC report wind turbing specific capacity classes used for determining costs
    rep_specific_capacities = [0.2,0.3,0.47] # 2018 JRC report wind turbine class representative specific capacities
    rep_hub_heights = {0.2:200, 0.3:100, 0.47:50} # 2018 JRC report wind turbine class hub heights for each of the three representative specific capacities
    iec_wind_classes = {1:15,2:8.5,3:7.5} # The wind clases (1-3, dictionary keys) are assigned a maximum wind speed (dictionary values) that can be sustained
    logger.info('Some code changed above. 10-->15 m/s')
    def __init__(self, turbine_type, onshore, classifications, component_specs, turbine_specs, hub_height=None,  wpl_kwargs={}):
        self.specific_capacity = turbine_specs.loc[turbine_specs.turbine_type==turbine_type,'nominal_power'].item()/1e3/turbine_specs.loc[turbine_specs.turbine_type==turbine_type,'rotor_area'].item() # [kW/m2]
        hub_heights_str = str(turbine_specs.loc[turbine_specs.turbine_type==turbine_type,'hub_height'].item()).replace(' ','')
        if hub_heights_str == 'nan': # Some turbines are missing hub heights. Assume a hub height providing 20m tip clearance
            self.hub_heights = [turbine_specs.loc[turbine_specs.turbine_type==turbine_type,'rotor_diameter'].item()/2 + 20]
        else:
            self.hub_heights = sorted([float(x) for x in re.split(';|/|,',hub_heights_str) if len(x)>0])
        self.iec_class = classifications.loc[classifications.turbine_type==turbine_type,'iec_class'].iloc[0]
        self.custom_power_curve = bool(classifications.loc[classifications.turbine_type==turbine_type,'custom_power_curve'].iloc[0])
        if hub_height==None:
            # Assigns a hub height to align most closely with the JRC representative types
            self.rep_specific_capacity = min(self.rep_specific_capacities, key=lambda x:abs(x-self.specific_capacity)) # Selects the JRC representative specific capacity closest to that of this turbine
            self.specific_capacity_class = self.specific_capacity_classes[self.rep_specific_capacity]
            self.rep_hub_height = self.rep_hub_heights[self.rep_specific_capacity] # Selects the representative hub height for the JRC wind turbine class closets to this turbine's specific capacity
            self.jrc_hub_height = min(self.hub_heights, key=lambda x:abs(x-self.rep_hub_height)) # [m] Selects the hub height for this model that is closest to the representative hub height found above
        if self.custom_power_curve:
            custom_power_curves = pd.read_csv(os.path.join(os.path.dirname(sys.argv[0]),'custom_power_curves.csv'),index_col=0) # Reads a csv file with custom power curves for some turbines
            cpc = custom_power_curves.loc[[turbine_type]].copy()
            cpc.dropna(axis=1,inplace=True)
            cpc = cpc.transpose().drop('Source').reset_index()
            cpc.columns = ['wind_speed','value']
            cpc = cpc.astype('float')
            wpl_kwargs['power_curve'] = cpc
            logger.info(f'Custom power curve for {turbine_type} loaded.')
        super().__init__(turbine_type=turbine_type, hub_height=self.jrc_hub_height, **wpl_kwargs)
        if onshore:
            if classifications.loc[classifications.turbine_type==turbine_type,'onshore'].iloc[0] != 1:
                raise Exception(f'This wind turbine model ({turbine_type}) does not have an onshore variant according to the classification.csv file.')
            self.costs = Component('wind',specs=component_specs,wind_class=self.specific_capacity_class)   
        else:
            if classifications.loc[classifications.turbine_type==turbine_type,'offshore'].iloc[0] != 1:
                raise Exception(f'This wind turbine model ({turbine_type}) does not have an offshore variant according to the classification.csv file.')
            self.monopole_costs = Component('wind',specs=component_specs,wind_class='monopole_costs') 
            self.floating_costs = Component('wind',specs=component_specs,wind_class='floating_costs') 
        
    def assign_foundation_costs(self,foundation):
        if foundation=='monopole':
            self.costs = self.monopole_costs
        elif foundation=='floating':
            self.costs = self.floating_costs
        else:
            raise Exception(f'Invalid offshore wind turbine foundation type: {foundation}.')

    def reset_hub_height(self):
        self.hub_height = self.jrc_hub_height
        
    def assign_site(self,v_50m,hellmann):
        '''Attempts to assign a hub height to the turbine model that complies with the IEC classification.
        Returns True is the assignment is successful.
        Returns False if it is found that the wind speeds are too high for even the lowest hub height. In this case the representative hub height is assigned to the model.
        '''
        self.reset_hub_height()
        hub_speed = v_50m*(self.hub_height/50)**hellmann
        mean_hub_speed = hub_speed.mean()
        if self.hub_height in self.hub_heights:
            hub_height_idx = self.hub_heights.index(self.hub_height)
        else:
            logger.error(f'A custom hub height ({self.hub_height}) was assigned to the turbine ({self.turbine_type}). This is being overridden.')
            hub_height_idx = len(self.hub_heights)-1
        feasible = False
        while hub_height_idx >= 0:
            if mean_hub_speed > self.iec_wind_classes[self.iec_class] and hub_height_idx > 0:
                hub_height_idx -= 1
                self.hub_height = self.hub_heights[hub_height_idx]
                hub_speed = v_50m*(self.hub_height/50)**hellmann
                mean_hub_speed = hub_speed.mean()
            elif mean_hub_speed > self.iec_wind_classes[self.iec_class] and hub_height_idx == 0:
                feasible = False
                self.reset_hub_height()
                break
            else:
                feasible = True
                break
        return feasible

## Generate wind turbine objects
turbines = {'onshore':{},'offshore':{}}
component_specs = pd.read_excel(os.path.join(SAF_directory,'data','plant_assumptions.xlsx'),sheet_name='data',index_col=0)
turbine_specs = pd.read_csv(os.path.join(os.path.dirname(wpl.__file__),'oedb','turbine_data.csv'))
classifications = pd.read_csv(os.path.join(os.path.dirname(sys.argv[0]),'classification.csv')) # Reads a csv that indicates the on-offshore and IEC classification of wind turbines
all_types = list(wpl.wind_turbine.get_turbine_types('local',filter_=True,print_out=False)['turbine_type'].unique())
invalids = [] #['SWT142/3150','SWT113/2300','SWT130/3300','S152/6330'] # The power curves for these turbines appear unrealistic
for x in invalids:
    all_types.remove(x)
onshore_types = [x for x in all_types if x in list(classifications.loc[classifications.onshore==1,'turbine_type'])] # List of on-shore turbine models
offshore_types = [x for x in all_types if x in list(classifications.loc[classifications.offshore==1,'turbine_type'])] # List of off-shore turbine models
missing_shore_designation = [x for x in all_types if x not in classifications.turbine_type.unique()]
if len(missing_shore_designation) > 0:
    logger.error(f'The following wind turbine types were not assigned on- or off-shore designations in the classifications.csv file. They are therefore not being used: {missing_shore_designation}')
loading_errors = {}
for model in all_types:
    try:
        if model in onshore_types:
            turbines['onshore'][model] = Turbine(turbine_type=model,onshore=True,classifications=classifications,component_specs=component_specs,turbine_specs=turbine_specs)
        if model in offshore_types: # Note that we are assuming ALL offshore wind turbines are medium-distance to shore, jacket base types
            turbines['offshore'][model] = Turbine(turbine_type=model,onshore=False,classifications=classifications,component_specs=component_specs,turbine_specs=turbine_specs)
    except Exception as e:
        loading_errors[model] = e
if len(loading_errors)>0:
    logger.info('There was a problem loading the following wind turbine models: {}'.format(str(loading_errors.keys())))
logger.info('The following onshore wind turbine models were properly loaded: {}'.format(str(turbines['onshore'].keys())))
logger.info('The following offshore wind turbine models were properly loaded: {}'.format(str(turbines['offshore'].keys())))

# Validate countries in config file. Remove countries not found in the MERRA folder
countries_filepath = os.path.join(os.path.dirname(sys.argv[0]),f'{script_name}.csv')

try:
    countries = list(pd.read_csv(countries_filepath, index_col=0)['country'])
except FileNotFoundError:
    raise Exception('Configuration file not found. The csv file containing the list of coutries to analyze should be found in the same directory location as this script.')

removed_countries = []
for country in countries:
    if country not in str(os.listdir(os.path.join(SAF_directory,'data','MERRA'))):
        removed_countries.append(country)
for country in removed_countries:
    countries.remove(country)
if len(removed_countries)>0:
    logger.info('The following countries were not found in the MERRA folder: {}'.format(removed_countries))
logger.info('Country set used for analysis: {}'.format(countries))

# Read evaluation points file used to filter out points that do not need to be evaluated
try:
    with open(os.path.join(SAF_directory,'data','Countries_WGS84/processed/Europe_MERRA_Evaluation_Points.json'),'r') as fp:
        merra_points_dict = json.load(fp)
    europe_merra_points = gpd.read_file(os.path.join(SAF_directory,'data','Countries_WGS84/processed/Europe_MERRA_Evaluation_Points.shp')) 
except FileNotFoundError as e:
    raise Exception(f'{e.filename} file not found. This file containing the MERRA points within each country\'s borders must be available at data/Countries_WGS84/processed/.')

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
        

def files_to_dataframe(country):
    '''Extracts all files in the country folder and concatenates them by date into a DataFrame.'''
    file_path = os.path.join(SAF_directory,'data','MERRA',country, '*.nc4')
    try:
        # with xr.open_mfdataset(file_path, concat_dim='date', combine='nested',preprocess=extract_date) as ds_wind:
        with xr.open_mfdataset(file_path, concat_dim='lat', combine='by_coords') as ds_wind:
            df_wind = ds_wind.to_dataframe()

    except xr.MergeError as e:
        logger.error('Merge error for {}:'.format(country))
        logger.error(e)
    except FileNotFoundError as e:
        logger.error(f'Process {os.getpid} error: {e}')
        exit()
    except Exception as e:
        logger.error(f'Process {os.getpid} error: {e}')
        exit()
        
    init_len = len(df_wind)
    df_wind.dropna(subset=['V10M','U10M','V50M','U50M'],how='any',inplace=True)
    if len(df_wind) < init_len:
        logger.info('{}: {} NaN rows were found and dropped.'.format(country,init_len - len(df_wind)))
    return df_wind

def preprocess_df(df,country):
    '''Resets and configures multi-index of dataframe, calculates wind velocities, and calculates Hellmann exponent.'''
    df_wind = df.copy()
    df_wind.reset_index(inplace=True)
    df_wind.loc[abs(df_wind.lon)<1e-3,'lon'] = 0 # xarray reads zeros as very small numbers, this sets them correctly
    df_wind.loc[abs(df_wind.lat)<1e-3,'lat'] = 0 # xarray reads zeros as very small numbers, this sets them correctly
    df_wind.time = pd.to_datetime(df_wind.time)
    # df_wind.drop(columns='date',inplace=True)
    df_wind.set_index(['lat','lon','time'],inplace=True)
    df_wind.sort_index(inplace=True)

    # Drop points not in "coast_points" or "land_points"
    df_len = len(df_wind) #Store the length for comparison
    # df_wind.drop(df_wind.loc[~df_wind.index.droplevel(2).isin([(x[1],x[0]) for x in eval_points[country]])].index,inplace=True)
    # eval_points = land_points.get(country,[]) + coast_points.get(country,[])
    eval_points = merra_points_dict[country]
    df_wind = df_wind.loc[df_wind.index.droplevel(2).isin(eval_points)]
    logger.info(f'Dropped {(df_len-len(df_wind))/df_len*100:.1f}% of points due to location outside {country} evaluation set.')

    # Calculate the wind speed from the northward and eastward velocity components
    df_wind['v_10m'] = np.sqrt(df_wind['U10M']**2 + df_wind['V10M']**2) #[m/s]
    df_wind['v_50m'] = np.sqrt(df_wind['U50M']**2 + df_wind['V50M']**2) #[m/s]

    # Calculate the "Hellmann exponent", which effectively describes the instability of the atmosphere depending on surface roughness, obstacles, etc.
    # The 50 meter wind speed corresponds to that 50 m above the ground, the 10 m wind speed corresponds to that 10 m above the zero-plane displacement height (DISPH) (Mosshammer, 2016)
    df_wind['hellmann'] = (np.log(df_wind.v_50m) - np.log(df_wind.v_10m)) / (np.log(50) - np.log(10 + df_wind.DISPH)) # (Mosshammer, 2016)
    df_wind.drop(columns=['U10M','V10M','U50M','V50M'],inplace=True)
    return df_wind

def assign_turbine_model(df,optimization_metric='lcoe',shore_designation='onshore',foundation_type=None):
    '''Calculates hypothetical power output for each turbine model in "models" and returns the model with the most optimal optimization metric value.
    
    Optimization can be performed with any one the following "optimization_metrics":
    "lcoe": levelized cost of electricity 
    "flh": full load hours
    "density": the power production density measured in kWh per land area required for the turbine
    '''
    turbine_spacing = 5 # meters of spacing per meter of rotor diameter [Bryer, 2012]
    winning_turbine = None
    if optimization_metric.lower() == 'lcoe':
        winning_value = 100
    else:
        winning_value = 0
    winning_output = None
    for turbine in turbines[shore_designation].values():
        feasible_model = turbine.assign_site(df.v_50m,df.hellmann)
        if not feasible_model:
            continue
        speed_at_hub = df.v_50m*(turbine.hub_height/50)**df.hellmann
        output = wpl.power_output.power_curve(speed_at_hub,turbine.power_curve.wind_speed,turbine.power_curve.value)
        output_sum = sum(output)/1e3 #kWh
        if optimization_metric.lower() == 'lcoe':
            if shore_designation=='offshore':
                turbine.assign_foundation_costs(foundation_type)
            metric = costs_NPV(capex=turbine.costs.CAPEX, opex=turbine.costs.OPEX,
                                discount_rate=component_specs.at['discount_rate', 'value'], lifetime=turbine.costs.lifetime,
                                capacity=turbine.nominal_power / 1e3) / np.sum([output_sum/(1+component_specs.at['discount_rate', 'value'])**n for n in np.arange(turbine.costs.lifetime+1)])  # EUR/kWh
            winner = metric < winning_value
        elif optimization_metric.lower() == 'flh':
            metric = output_sum/(turbine.power_curve.value.max()/1e3)
            winner = metric > winning_value
        elif optimization_metric.lower() == 'density':
            turbine_area = (turbine.rotor_diameter*turbine_spacing)**2 # [m^2]
            metric = output_sum/turbine_area #kWh/m2
            winner = metric > winning_value
        else:
            logger.error('Invalid wind turbine optimization_metric. Must be "lcoe", "flh", or "density".')
            sys.exit()
        if winner:
            winning_turbine = turbine
            winning_value = metric
            winning_output = output
    if winning_turbine == None:
        refnum = os.getpid()
        logger.error(f'Problem assigning a turbine model. Latest metric value {metric}. DataFrame saved to the cache with reference: {refnum}.')
        df.to_csv(os.path.join(cache_path,f'{refnum}.csv'))
    return winning_turbine, winning_output
        
# def compute_power_output(df,offshore_points=None):
def compute_power_output(df):
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
    df['rep_hub_height'] = ''
    df['hub_height'] = ''
    df['offshore'] = ''
    for coords in df.index.droplevel(2).unique():
        # Get shore designation
        is_offshore = europe_merra_points.loc[(europe_merra_points.grid_lat==coords[0])&(europe_merra_points.grid_lon==coords[1]),'pt_in_sea'].iloc[0]
        shore_designation = 'offshore' if is_offshore else 'onshore'
            if is_offshore:
                shore_dist = europe_merra_points.loc[(europe_merra_points.grid_lat==coords[0])&(europe_merra_points.grid_lon==coords[1]),'shore_dist'].max()
                foundation_type = 'monopile' if shore_dist <=60 else 'floating'
            else:
                foundation_type=None
        turbine,output = assign_turbine_model(df.loc[coords],optimization_metric=optimization_metric,shore_designation=shore_designation,foundation_type=foundation_type)
        # turbine = turbines[shore_designation][optimal_turbine]
        # speed_at_hub = df.loc[coords].v_50m*(turbine.hub_height/50)**df.loc[coords].hellmann
        # output = wpl.power_output.power_curve(speed_at_hub,turbine.power_curve.wind_speed,turbine.power_curve.value) # [Wh]
        df.loc[idx[coords],'kWh'] = list(output/1e3) # [kWh]
        df.loc[idx[coords],'turbine_type'] = turbine.turbine_type
        df.loc[idx[coords],'rated_power_MW'] = turbine.nominal_power/1e6 # [MW]
        df.loc[idx[coords],'rotor_diameter'] = turbine.rotor_diameter # [m]
        df.loc[idx[coords],'specific_capacity_class'] = turbine.specific_capacity_class
        df.loc[idx[coords],'rep_hub_height'] = turbine.jrc_hub_height # [m]
        df.loc[idx[coords],'hub_height'] = turbine.hub_height # [m]
        df.loc[idx[coords],'offshore'] = bool(is_offshore) # [m]

def process_country(country):
    '''Extract files to DataFrame, preprocess dataframe, and compute power output for each country in the "countries" list'''
    # Create a script_name for caching the interim results: "01_merra_wind_preprocessing_<country>_<script_time>.csv"
    logger.info(f'{country} processing started...')
    cache_file_name = f'{script_name}_{country}_{sstime_string}.csv'
    # Extract files & preprocess DataFrame
    df_wind = files_to_dataframe(country)
    df_wind = preprocess_df(df_wind,country)
    # Cache DatFrame
    logger.info(f'Initial caching for {country}...')
    df_wind.to_parquet(os.path.join(cache_path,f'{cache_file_name}.parquet.gzip'),compression='gzip')
    logger.info(f'Initial {country} cache saved')
    # # ADD SOME CODE HERE TO IDENTIFY POINTS THAT ARE OFFSHORE
    # logger.error(f'The points in {country} are not assigned an on/offshore designation. Thus, all points are assumed to be onshore.')
    # offshore_points = coast_points.get(country,[])
    # Calculate power output
    compute_power_output(df_wind)
    # Cache & save results
    logger.info(f'Saving results for {country}...')
    df_wind.to_parquet(os.path.join(cache_path,f'{cache_file_name}.parquet.gzip'),compression='gzip')
    df_wind.to_parquet(os.path.join(results_path,f'{country}.parquet.gzip'),compression='gzip')
    # df_wind.to_csv(os.path.join(results_path,f'{country}.csv'))
    logger.info(f'Results for {country} saved')

# for country in countries:
#     process_country(country)

cores_avail = mp.cpu_count()
logger.info(f'{cores_avail} cores available')
P = mp.Pool(min(cores_avail,max_processes,len(countries)))
P.map(process_country,countries)
P.close()
P.join()

logger.info('Script time: {:.2f} seconds.'.format(time.time()-sstime))