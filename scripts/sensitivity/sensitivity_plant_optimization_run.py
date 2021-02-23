# python scripts/sensitivity/sensitivity_plant_optimization_run.py -d . -n 1 -s 10

import os
import sys
import numpy as np
import argparse
import time
import pandas as pd
import random

desc_text = 'This script submits jobs to run the sensitivity analysis.'
parser = argparse.ArgumentParser(description=desc_text)
parser.add_argument('-d','--SAF_directory',
    help='The path to the "SAFlogistics" directory',
    default='$HOME/EuroSAFs',
    type=str)
parser.add_argument('-o','--offshore',
    action='store_true',
    help='Calculate offshore points.',
    default=False)
parser.add_argument('-n','--bin_size',
    help='The maximum number of points that will be evaluated for each job.',
    default=50,
    type=int)
parser.add_argument('-s','--sample_size',
    help='The number of sensitivity points that will be evaluated',
    default=500,
    type=int)
args = parser.parse_args()

SAF_directory = args.SAF_directory
offshore = args.offshore
onshore = not offshore
bin_size = args.bin_size
sample_size = args.sample_size

if offshore:
    results_path = os.path.join(results_path,'offshore')
    offshore_flag = '--offshore'
else:
    offshore_flag = ''

cores = 32
wall_time = '30:00'
MIPGap = 0.01
DisplayInterval = 30

if SAF_directory == None:
    SAF_directory = os.environ['HOME']
    SAF_directory = os.path.join(SAF_directory,'EuroSAFs')
results_path = os.path.join(SAF_directory,'results','sensitivity')
if not os.path.isdir(results_path):
    os.mkdir(results_path)

europe_points = pd.read_csv(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_Evaluation_Points.csv'),index_col=0)
points_set = europe_points.loc[europe_points.sea_node==offshore]

rand_idxs = random.sample(list(points_set.index),sample_size)
eval_points = points_set.loc[rand_idxs].sort_index()
if offshore:
    eval_points.to_csv(os.path.join(SAF_directory,'scripts/sensitivity/eval_points_offshore.csv'))
else:
    eval_points.to_csv(os.path.join(SAF_directory,'scripts/sensitivity/eval_points_onshore.csv'))
points = list(eval_points.index.unique())
time.sleep(1)

# loop through the countries list: 
bins = int(np.ceil(len(eval_points)/bin_size))
for i in range(bins):
    i+=1
    bash_str = f'bsub -n {cores} -W {wall_time} -J "sensitivity-{i}" -oo {results_path}/lsf.sensitivity-{i}.txt '\
            f'python $HOME/EuroSAFs/scripts/03_plant_optimization/02_plant_optimization.py '\
            f'--SAF_directory {SAF_directory} '\
            f'--country sensitivity '\
            f'--year 2020 '\
            f'--bin_number {i} '\
            f'--bin_size {bin_size} '\
            f'--MIPGap {MIPGap} '\
            f'--DisplayInterval {DisplayInterval} '\
            f'{offshore_flag}'\
            f'--save_operation '\
            f'--verbose '\
            f'--sensitivity_analysis'
    os.system(bash_str)
    time.sleep(0.01)