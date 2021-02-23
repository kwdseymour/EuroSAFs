# python scripts/sensitivity/plant_optimization.py -d . -m 0.01 -v -s


import sys
import argparse
import os
import random
import logging
import pandas as pd
import numpy as np
import plant_optimization as pop

sstime = time.time()
sstime_string = datetime.datetime.fromtimestamp(sstime).strftime('%d%m%Y%H')
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

desc_text = 'This script runs the plant optimizer for every location in the given country.'
parser = argparse.ArgumentParser(description=desc_text)
parser.add_argument('-d','--SAF_directory',
    help='The path to the "SAFlogistics" directory',)
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
    default=2000,
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
args = parser.parse_args()

MIPGap = args.MIPGap
DisplayInterval = args.DisplayInterval
silent_optimizer = not(args.verbose_optimizer)
save_operation = args.save_operation
timelimit = args.time_limit
bin_number = args.bin_number
offshore = args.offshore
onshore = not offshore
bin_size = args.bin_size
max_processes = args.max_processes
SAF_directory = args.SAF_directory
if 'cluster/home' in os.getcwd():
    scratch_path = os.environ['SCRATCH']
else:
    scratch_path = os.path.join(SAF_directory,'scratch')

# Define path to logs, cache, and results folders. Create directories if they don't already exist
logs_path = os.path.join(scratch_path,'logs')
if not os.path.isdir(logs_path):
    os.mkdir(logs_path)
cache_path = os.path.join(scratch_path,'cache')
if not os.path.isdir(cache_path):
    os.mkdir(cache_path)
results_path = os.path.join(SAF_directory,'results',script_name)
if not os.path.isdir(results_path):
    os.mkdir(results_path)
if save_operation:
    operations_path = os.path.join(results_path,'operation')
    if not os.path.isdir(operations_path):
        os.mkdir(operations_path)
eval_points_path = os.path.join(results_path,'eval_points.txt')

# Add a logger
sys.path.insert(1, os.path.join(SAF_directory,'scripts','03_plant_optimization'))
from plant_optimization.utilities import create_logger
logger = create_logger(scratch_path,__name__,__file__)

# Add the file handlers from this script's logger to the plant_optimization.plant_optimizer logger so they are all printed to this script's log
popt_logger = logging.getLogger('plant_optimization.plant_optimizer')
popt_logger.addHandler(logger.handlers[0])
popt_logger.addHandler(logger.handlers[1])

cores_avail = multiprocessing.cpu_count()
logger.info(f'{cores_avail} cores available')
if max_processes == None:
    max_processes = cores_avail - 1
    logger.info(f'Max processes parameter not set. Using {max_processes} cores.')


if offshore:
    points = pd.read_csv(os.path.join(SAF_directory,'scripts/sensitivity/eval_points_offshore.csv'),index_col=0)
else:
    points = pd.read_csv(os.path.join(SAF_directory,'scripts/sensitivity/eval_points_onshore.csv'),index_col=0)
points = list(points.index.unique())

bin_count = int(np.ceil(len(points)/bin_size))
bin_string = f'_{bin_number}-{bin_count}'
points = points[(bin_number-1)*bin_size:bin_number*bin_size]

if os.path.isfile(os.path.join(results_path,'sensitivity'+bin_string+'.csv'))):
    logger.error(f'Sensitivity results already found in results folder. Remove file in order to perform new analysis.')
    sys.exit()

results_df = pd.DataFrame()
for i,row in points.iterrows():
    country = row.country
    point = (row.lat,row.lon)
    logger.info(f'Starting point {i} of {len(points)} ({country}).')
    try:
        site = pop.Site(point,country,offshore=offshore)
        plant = pop.Plant(site)
        pop.optimize_plant(plant,threads=max_processes,MIPGap=MIPGap,timelimit=timelimit,DisplayInterval=DisplayInterval,silent=silent_optimizer)
        try:
            pop.unpack_design_solution(plant, unpack_operation=True)
        except:
            logger.error(f'There was a problem unpacking the optimizer model for point({point}) in {country}.')
            raise Exception('1')
    except pop.errors.CoordinateError:
        logger.error(f'Coordinate error for point ({point}) in {country}.')
        continue
    except:
        logger.error(f'Optimization problem for point ({point}) in {country}.')
        raise Exception('2')
    try:
        results_dict = pop.solution_dict(plant)
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

    if save_operation:
        plant.operation.to_parquet(os.path.join(operations_path,f'{country}_{site.lat}_{site.lon}_{site.shore_designation}.parquet.gzip'),compression='gzip')

    results_df = results_df.append(results_dict,ignore_index=True)

results_df = results_df[['lat','lon','shore_designation','turbine_type','rotor_diameter','rated_turbine_power','wind_turbines',
'wind_capacity_MW','PV_capacity_MW','electrolyzer_capacity_MW','CO2_capture_tonph','heatpump_capacity_MW',
'battery_capacity_MWh','H2stor_capacity_MWh','CO2stor_capacity_ton','H2tL_capacity_MW','curtailed_el_MWh',
'wind_production_MWh','PV_production_MWh','NPV_EUR','CAPEX_EUR','LCOF_MWh','LCOF_liter','runtime']]

if offshore:
    results_path = os.path.join(results_path,'offshore')
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
results_df.to_csv(os.path.join(results_path,country+bin_string+'.csv'))

logger.info(f'Script finished after {time.time()-sstime:.1f} seconds.')