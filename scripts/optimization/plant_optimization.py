#!~/anaconda3/envs/GIS/bin/python
# coding: utf-8

# python scripts/optimization/plant_optimization.py -d . -c Austria -m 0.01 -v

'''
This script runs the plant optimizer for every location in the given country.

----- File Inputs -----
- wind data: preprocessed wind power output data (parquet) files for each country should be made available in the following path relative to the "EuroSAFs" directory:
    "/results/wind_power_output/"

- PV data: PV output data (parquet) files for each country should be made available in the following path relative to the "EuroSAFs" directory:
    "/results/PV_power_output/"

- plant assumptions: an updated plant_assumptions.xlsx file must be available in the following path relative to the "EuroSAFs" directory:
    "/data/plant_assumptions.xlsx"


----- File Outputs -----
- logs: logging files are written to the following path relative to the "scratch" directory:
    "scratch/logs/"
If this script is run on an Euler home directory, it will locate the "scratch" path via the $SCRATCH shell environment variable. Otherwise it must be located in the "EuroSAFs" directory.

- evaluation points: this text file contains a line for each evaluated point with a message indicating its success or reason for error. 
It is saved in the following path relative to the "EuroSAFs" directory:
    "/results/{script_name}/eval_points.txt

- optimization results: this CSV file contains the results of the plant optimization at each point in the given country.
It is saved in the following path relative to the "EuroSAFs" directory:
    "/results/{script_name}/{country}.csv
'''

import sys
import argparse
import os
import time
import datetime
import logging
import pandas as pd
import numpy as np
import multiprocessing
import plant_optimization as pop

sstime = time.time()
sstime_string = datetime.datetime.fromtimestamp(sstime).strftime('%d%m%Y%H')
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

desc_text = 'This script runs the plant optimizer for every location in the given country.'
parser = argparse.ArgumentParser(description=desc_text)
parser.add_argument('-d','--EuroSAFs_directory',
    help='The path to the "EuroSAFs" directory',)
parser.add_argument('-r','--results_path',
    help='The path to the directory where the results should be stored',
    type=str)
parser.add_argument('-c','--country',
    help='The country for which to run the optimization.',
    type=str)
parser.add_argument('-y','--year',
    help='Select which year to select plant assumptions from. Default is 2020.',
    default=2020,
    type=int)
parser.add_argument('-p','--max_processes',
    help='Controls the number of threads to apply to parallel algorithms (of the optimizer)',
    default=None)
parser.add_argument('-n','--bin_size',
    help='The maximum number of points that will be evaluated for the given country.',
    type=int)
parser.add_argument('-b','--bin_number',
    help='Indicates which bin of coordinates from the country to evaluate',
    default=1,
    type=int)
parser.add_argument('-t','--time_limit',
    help='Sets the time limit of the optimizer (seconds)',
    default=3000,
    type=int)
parser.add_argument('-m','--MIPGap',
    help='Sets the MIP gap of the optimizer',
    default=0.0001,
    type=float)
parser.add_argument('-i','--DisplayInterval',
    help='Sets the time interval of the Gurobi logger display if silent is set to False',
    default=10,
    type=float)
parser.add_argument('-v','--verbose_optimizer',
    action='store_true',
    help='Prints the optimizer output.',
    default=False)
parser.add_argument('-s','--save_operation',
    action='store_true',
    help='Saves the plant operation data to an operations folder in the results folder.',
    default=False)
parser.add_argument('-o','--offshore',
    action='store_true',
    help='Calculate offshore points.',
    default=False)
parser.add_argument('-a','--sensitivity_analysis',
    action='store_true',
    help='Generate random paramters for each plant.',
    default=False)
args = parser.parse_args()

MIPGap = args.MIPGap
DisplayInterval = args.DisplayInterval
silent_optimizer = not(args.verbose_optimizer)
save_operation = args.save_operation
timelimit = args.time_limit
bin_number = args.bin_number
offshore = args.offshore
onshore = not offshore
sensitivity = args.sensitivity_analysis
year = args.year
bin_size = args.bin_size
country = args.country
max_processes = args.max_processes
EuroSAFs_directory = args.EuroSAFs_directory
results_path = args.results_path

# Define the path to the scratch directory. This is where logging files will be saved
if 'cluster/home' in os.getcwd():
    scratch_path = os.environ['SCRATCH']
else:
    scratch_path = os.path.join(EuroSAFs_directory,'scratch')
# Create a directory if it doesn't already exist
if not os.path.isdir(scratch_path):
    os.makedires(scratch_path)

# Create a results directory if it doesn't already exist
if not os.path.isdir(results_path):
    os.makedires(results_path)

# Create a directory to save plant operation data
if save_operation:
    operations_path = os.path.join(results_path,'operation')
    if not os.path.isdir(operations_path):
        os.mkdir(operations_path)
eval_points_path = os.path.join(results_path,'eval_points.txt')

# Add a logger
sys.path.insert(1, os.path.join(EuroSAFs_directory,'scripts','optimization'))
from plant_optimization.utilities import create_logger
logger = create_logger(scratch_path,__name__,__file__)

# Add the file handlers from this script's logger to the plant_optimization.plant_optimizer logger so they are all printed to this script's log
popt_logger = logging.getLogger('plant_optimization.plant_optimizer')
popt_logger.addHandler(logger.handlers[0])
popt_logger.addHandler(logger.handlers[1])

# Set the maximum processes parameter according to available cores if parameter not already set
cores_avail = multiprocessing.cpu_count()
logger.info(f'{cores_avail} cores available')
if max_processes == None:
    max_processes = cores_avail - 1
    logger.info(f'Max processes parameter not set. Using {max_processes} cores.')

# Extract the evaluation points 
if sensitivity:
    sensitivity_points = pd.read_csv(os.path.join(results_path,'eval_points.csv'),index_col=0)
    eval_points = sensitivity_points.set_index(['grid_lat','grid_lon'])
else:
    europe_points = pd.read_csv(os.path.join(EuroSAFs_directory,'data/Countries_WGS84/processed/Europe_Evaluation_Points.csv'),index_col=0)
    eval_points = europe_points.loc[europe_points.country==country].set_index(['grid_lat','grid_lon'])

if onshore:
    eval_points = eval_points.loc[~eval_points.sea_node]
else:
    eval_points = eval_points.loc[eval_points.sea_node]

# Divide the evaluation points into bins according to the bin_size specified as a script argument
bin_count = int(np.ceil(len(eval_points)/bin_size))
bin_string = f'_{bin_number}-{bin_count}' # This is used for file handling purposes
# Extract the slice of evaluation points according to the bin_number specified as a script argument
points_slice = eval_points.iloc[(bin_number-1)*bin_size:bin_number*bin_size]

# Verify that the result doesn't already exist. Exit if it does.
if os.path.isfile(os.path.join(results_path,country+bin_string+'.csv')):
    logger.error(f'{country} results already found in results folder. Remove file in order to perform new analysis.')
    sys.exit()

# Initialize a df to hold the results
results_df = pd.DataFrame()

# Loop through every point and perform the plant optimization on each
points = list(points_slice.index)
for i,point in enumerate(points):
    # with open(eval_points_path,'r') as fp:
    #     eval_points_str = fp.read()
    # if str(point) not in eval_points_str:
    logger.info(f'Starting {country} point {i+1} of {len(points)}.')
    try:
        if sensitivity:
            eval_country = points_slice.iloc[i]['country']
        else:
            eval_country = country

        # Initialize the plant site, which reads the wind & PV power output data
        site = pop.Site(point,eval_country,offshore=offshore)

        # Initialize a plant object, which holds all the component specifications/costs
        plant = pop.Plant(site=site,year=year,sensitivity=sensitivity)

        # Run the plant optimizer
        pop.optimize_plant(plant,threads=max_processes,MIPGap=MIPGap,timelimit=timelimit,DisplayInterval=DisplayInterval,silent=silent_optimizer)

        # Unpack the optimization solution
        try:
            pop.unpack_design_solution(plant, unpack_operation=True)
        except:
            logger.error(f'There was a problem unpacking the optimizer model for point({point}) in {country}.')
            continue

    except pop.errors.CoordinateError:
        logger.error(f'Coordinate error for point ({point}) in {country}.')
        continue
    except:
        logger.error(f'Optimization problem for point ({point}) in {country}.')
        continue

    # Extract the plant solution to a dictionary
    try:
        results_dict = pop.solution_dict(plant)
        
        # Extract the plant specifications and add it to the solutions dictionary 
        results_dict.update(dict(plant.specs.value))
        
        results_dict['country'] = eval_country
        with open(eval_points_path,'a') as fp:
            fp.write(f'\n{country} point {point}: success.')
    except pop.plant_optimizer.CoordinateError as e:
        logger.error(e.__str__())
        with open(eval_points_path,'a') as fp:
            fp.write(f'\n{country} point {point}: {e.__str__()}.')
    except pop.plant_optimizer.OptimizerError as e:
        logger.error(e.__str__())
        with open(eval_points_path,'a') as fp:
            fp.write(f'\n{country} point {point}: {e.__str__()}.')
    except Exception as e:
        logger.error(f'There was a problem saving the optimizer results for point({point}) in {country}: {e}')
        with open(eval_points_path,'a') as fp:
            fp.write(f'\n{country} point {point}: {e}')

    if save_operation: # Save the plant operational data
        plant.operation.to_parquet(os.path.join(operations_path,f'{country}_{site.lat}_{site.lon}_{site.shore_designation}.parquet.gzip'),compression='gzip')

    # Append the results to the results df
    results_df = results_df.append(results_dict,ignore_index=True)

# results_df = results_df[['lat','lon','shore_designation','turbine_type','rotor_diameter','rated_turbine_power','wind_turbines',
# 'wind_capacity_MW','PV_capacity_MW','electrolyzer_capacity_MW','CO2_capture_tonph','boiler_capacity_MW',
# 'battery_capacity_MWh','H2stor_capacity_MWh','CO2stor_capacity_ton','H2tL_capacity_MW','curtailed_el_MWh',
# 'wind_production_MWh','PV_production_MWh','NPV_EUR','CAPEX_EUR','LCOF_MWh','LCOF_liter','runtime']]

results_df.to_csv(os.path.join(results_path,country+bin_string+'.csv'))

logger.info(f'Script finished after {time.time()-sstime:.1f} seconds.')