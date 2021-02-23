# python scripts/03_plant_optimization/02_plant_optimization_run.py
# python scripts/03_plant_optimization/02_plant_optimization_run.py -o

import os
import sys
import numpy as np
import argparse
import time
import pandas as pd

desc_text = 'This script submits jobs to run all optimizations for all countries.'
parser = argparse.ArgumentParser(description=desc_text)
parser.add_argument('-d','--SAF_directory',
    help='The path to the "SAFlogistics" directory',
    default = None,
    type=str)
parser.add_argument('-o','--offshore',
    action='store_true',
    help='Calculate offshore points.',
    default=False)
parser.add_argument('-y','--year',
    help='Select which year to select plant assumptions from. Default is 2020.',
    default=2020,
    type=int)
parser.add_argument('-n','--bin_size',
    help='The maximum number of points that will be evaluated for each job.',
    default=50,
    type=int)
args = parser.parse_args()

SAF_directory = args.SAF_directory
offshore = args.offshore
onshore = not offshore
year = args.year
bin_size = args.bin_size

cores = 32
wall_time = '30:00'
# cores = 8
# wall_time = '3:00'
MIPGap = 0.01
DisplayInterval = 30
if SAF_directory == None:
    SAF_directory = os.environ['HOME']
    SAF_directory = os.path.join(SAF_directory,'EuroSAFs')
results_path = os.path.join(SAF_directory,'results','02_plant_optimization')
if offshore:
    results_path = os.path.join(results_path,'offshore')
    offshore_flag = '--offshore'
else:
    offshore_flag = ''

europe_points = pd.read_csv(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_Evaluation_Points.csv'),index_col=0)
countries = europe_points.country.unique()
# countries = ['Spain']


# loop through the countries list: 
for country in countries:
    points = europe_points.loc[europe_points.country==country].set_index(['grid_lat','grid_lon'])
    if onshore and not offshore:
        points = points.loc[~points.sea_node]
    elif not onshore and offshore:
        points = points.loc[points.sea_node]
    elif onshore and offshore:
        pass
    else:
        print('Either onshore or offshore must be set to TRUE')
    points = list(points.index.unique())

    bins = int(np.ceil(len(points)/bin_size))
    
    
    for i in range(bins):
        i+=1
        bin_string = f'_{i}-{bins}'
        file_path = os.path.join(results_path,country+bin_string+'.csv')
        if os.path.isfile(file_path):
            print(f'{file_path} already exists. Delete file to run this set.')
            continue
            
        bash_str = f'bsub -n {cores} -W {wall_time} -J "{country}-{i}" -oo {results_path}/lsf.{country}-{i}.txt '\
            f'python $HOME/EuroSAFs/scripts/03_plant_optimization/02_plant_optimization.py '\
            f'--SAF_directory {SAF_directory} '\
            f'--country {country} '\
            f'--year {year} '\
            f'--bin_number {i} '\
            f'--bin_size {bin_size} '\
            f'--MIPGap {MIPGap} '\
            f'--DisplayInterval {DisplayInterval} '\
            f'{offshore_flag}'\
            f'--save_operation '\
            f'--verbose '\
        # bash_str = f'python $HOME/EuroSAFs/scripts/03_plant_optimization/02_plant_optimization.py -d $HOME/EuroSAFs -c {country} -m {MIPGap} -i {DisplayInterval} -b {i} -n {bin_size} -v -s'
        # bash_str = f'python $HOME/GitHub/EuroSAFs/scripts/03_plant_optimization/02_plant_optimization.py -d $HOME/GitHub/EuroSAFs -c {country} -m {MIPGap} -i {DisplayInterval} -b {i} -n {bin_size} -v -s'
        os.system(bash_str)
        time.sleep(0.1)