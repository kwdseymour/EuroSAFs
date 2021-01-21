#!~/anaconda3/envs/GIS/bin/python
# coding: utf-8

# python scripts/02_plant_optimization/02_plant_optimization.py -d . -c Austria -m 0.01 -v

'''
This script runs the plant optimizer for every location in the given country.

----- File Inputs -----
- wind data: preprocessed wind power output data (parquet) files for each country should be made available in the following path relative to the "SAFlogistics" directory:
    "/results/01_merra_wind_preprocessing/"

- PV data: PV output data (parquet) files for each country should be made available in the following path relative to the "SAFlogistics" directory:
    "/data/PVGIS/"

- plant assumptions: an updated plant_assumptions.xlsx file must be available in the following path relative to the "SAFlogistics" directory:
    "/data/plant_assumptions.xlsx"


----- File Outputs -----
- logs: logging files are written to the following path relative to the "scratch" directory:
    "scratch/logs/"
If this script is run on an Euler home directory, it will locate the "scratch" path via the $SCRATCH shell environment variable. Otherwise it must be located in the "SAFlogistics" directory.

- evaluation points: this text file contains a line for each evaluated point with a message indicating its success or reason for error.
It is saved in the following path relative to the "SAFlogistics" directory:
    "/results/{script_name}/eval_points.txt

- optimization results: this CSV file contains the results of the plant optimization at each point in the given country.
It is saved in the following path relative to the "SAFlogistics" directory:
    "/results/{script_name}/{country}.csv
'''

import sys
import argparse
import os
import time
import datetime
import logging
import plant_optimization as pop

sstime = time.time()
sstime_string = datetime.datetime.fromtimestamp(sstime).strftime('%d%m%Y%H')
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

desc_text = 'This script runs the plant optimizer for every location in the given country.'
parser = argparse.ArgumentParser(description=desc_text)
parser.add_argument('-c','--country',
    help='The country for which to run the optimization.',
    type=str)
parser.add_argument('-d','--SAF_directory',
    help='The path to the "SAFlogistics" directory',)
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
parser.add_argument('-points','--points',
    help='The point for which to run the optimization.',
    nargs='+')
parser.add_argument('-s','--source',
    help='The source for the wind data.',
    type=str)

args = parser.parse_args()
MIPGap = args.MIPGap
DisplayInterval = args.DisplayInterval
silent_optimizer = not(args.verbose_optimizer)
timelimit = args.time_limit
country = args.country
source = args.source
points = args.points
SAF_directory = args.SAF_directory

points = list(map(eval, points))[0]

# Define path to logs, cache, and results folders. Create directories if they don't already exist
logs_path = os.path.join(SAF_directory,'logs')
if not os.path.isdir(logs_path):
    os.mkdir(logs_path)
cache_path = os.path.join(SAF_directory,'cache')
if not os.path.isdir(cache_path):
    os.mkdir(cache_path)

results_pathes = {
                   'on_cost': os.path.join('.', 'results', '02_plant_optimization', 'optimal_cost', country),
                   'on_out': os.path.join('.', 'results', '02_plant_optimization', 'optimal_out', country),
                   'off_cost': os.path.join('.', 'results', '02_plant_optimization', 'optimal_cost', country, 'seaside'),
                   'off_out': os.path.join('.', 'results', '02_plant_optimization', 'optimal_out', country, 'seaside')
                  }

results_path = results_pathes[source]

eval_points_path = os.path.join(results_path,'eval_points.txt')

# Add a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(process)d - %(levelname)s: %(message)s','%Y-%m-%d %H:%M:%S')
file_handler1 = logging.FileHandler(os.path.join(SAF_directory,'logs','02_plant_optimization_persistent.log'))
file_handler1.setLevel(logging.INFO)
file_handler1.setFormatter(formatter)
file_handler2 = logging.FileHandler(os.path.join(SAF_directory,'logs','02_plant_optimization.log'),mode='w')
file_handler2.setLevel(logging.DEBUG)
file_handler2.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler1)
logger.addHandler(file_handler2)
logger.addHandler(stream_handler)
logger.propogate = False

# Add the file handlers from this script's logger to the plant_optimization.plant_optimizer logger so they are all printed to this script's log
popt_logger = logging.getLogger('plant_optimization.plant_optimizer')
popt_logger.addHandler(file_handler1)
popt_logger.addHandler(file_handler2)

def optimize(point, source):
    site = pop.Site(point, country, source)
    plant = pop.Plant(site)
    pop.optimize_plant(plant, eval_points_path, results_path, country, MIPGap=MIPGap,timelimit=timelimit,DisplayInterval=DisplayInterval,silent=silent_optimizer)

for point in points:
    optimize(point, source)