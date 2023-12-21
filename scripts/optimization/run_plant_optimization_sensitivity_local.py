# python scripts/optimization/run_plant_optimization_sensitivity.py -d $HOME/EuroSAFs -s 1000

import os
import numpy as np
import argparse
import time
import pandas as pd
import random
import git

desc_text = 'This script submits jobs to run the sensitivity analysis.'
parser = argparse.ArgumentParser(description=desc_text)
parser.add_argument('-d','--EuroSAFs_directory',
    help='The path to the "EuroSAFs" directory',
    default=None,
    type=str)
parser.add_argument('-o','--offshore',
    action='store_true',
    help='Calculate offshore points.',
    default=False)
parser.add_argument('-y','--year',
    help='Select which year to select plant assumptions from. Default is 2020.',
    default=2020,
    type=int)
parser.add_argument('-s','--sample_size',
    help='The number of sensitivity points that will be evaluated',
    default=50,
    type=int)
args = parser.parse_args()

EuroSAFs_directory = args.EuroSAFs_directory
offshore = args.offshore
sample_size = args.sample_size
year = args.year

save_operation = False
save_operation_flag = '--save_operation' if save_operation else ''

# Set job resource requirements
cores = 16 # Number of cores requested for each job
wall_time = '40:00' # Wall time requested for each job

# Set optimizer parameters
MIPGap = 0.01
DisplayInterval = 30 # This sets how frequently Gurobi prints the optimizer progress

# Define path to primary directory
if EuroSAFs_directory == None:
    EuroSAFs_directory = os.environ['HOME']
    EuroSAFs_directory = os.path.join(EuroSAFs_directory,'EuroSAFs')

# Retrieves the SHA code of the current git commit. This is used for file handling
try:
    repo = git.Repo(search_parent_directories=True)
    git_sha = repo.head.object.hexsha
    git_sha_str = git_sha[:7]+'_'
except:
    git_sha_str = ''


# Define the results path. The final results folder will contain the git SHA code plus the given evalutation year
results_path = os.path.join(EuroSAFs_directory,'results','plant_optimization',git_sha_str+'sensitivity')
# Further identify results path by oshore/offshore
if offshore:
    results_path = os.path.join(results_path,'offshore')
    offshore_flag = '--offshore'
else:
    results_path = os.path.join(results_path,'onshore')
    offshore_flag = ''
# Create the directory if it doesn't already exist
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# Get all evaluation points
europe_points = pd.read_csv(os.path.join(EuroSAFs_directory,'data/Countries_WGS84/processed/Europe_Evaluation_Points.csv'),index_col=0)
points_set = europe_points.loc[europe_points.sea_node==offshore]

# Randomly select evaluation points
rand_idxs = random.sample(list(points_set.index),sample_size)
eval_points = points_set.loc[rand_idxs].sort_index()

# Save randomly selected points to CSV file to be read by plant_optimization.py script
eval_points.to_csv(os.path.join(results_path,'eval_points.csv'))
points = list(eval_points.index.unique())
time.sleep(1)

# Generate the job submission string to be submitted to Euler
bash_str = f'python {EuroSAFs_directory}/scripts/optimization/plant_optimization.py '\
        f'--EuroSAFs_directory {EuroSAFs_directory} '\
        f'--results_path {results_path} '\
        f'--country sensitivity '\
        f'--year 2020 '\
        f'--bin_number 1 '\
        f'--bin_size {sample_size} '\
        f'--MIPGap {MIPGap} '\
        f'--DisplayInterval {DisplayInterval} '\
        f'{offshore_flag} '\
        f'{save_operation_flag} '\
        f'--verbose '\
        f'--sensitivity_analysis'

# Execute the generated job submission bash command 
os.system(bash_str)