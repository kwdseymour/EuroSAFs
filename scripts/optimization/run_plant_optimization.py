# python scripts/optimization/run_plant_optimization.py
# python scripts/optimization/run_plant_optimization.py -o

import os
import sys
import numpy as np
import argparse
import time
import pandas as pd
import git

desc_text = 'This script submits jobs to Euler to run all optimizations for all countries.'
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

# Set job resource requirements
cores = 32 # Number of cores requested for each job
wall_time = '30:00' # Wall time requested for each job

# Set optimizer parameters
MIPGap = .1
DisplayInterval = 30 # This sets how frequently Gurobi prints the optimizer progress

# Define path to primary directory
if SAF_directory == None:
    SAF_directory = os.environ['HOME']
    SAF_directory = os.path.join(SAF_directory,'EuroSAFs')

# Retrieves the SHA code of the current git commit. This is used for file handling
try:
    repo = git.Repo(search_parent_directories=True)
    git_sha = repo.head.object.hexsha
    git_sha_str = git_sha[:7]+'_'
except:
    git_sha_str = ''

# Define the results path. The final results folder will contain the git SHA code plus the given evaluation year
results_path = os.path.join(SAF_directory,'results','plant_optimization',git_sha_str+str(year))
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
    time.sleep(1)

# Get all evaluation points
europe_points = pd.read_csv(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_Evaluation_Points.csv'),index_col=0)
countries = europe_points.country.unique()
# countries = ['Switzerland']


# loop through the countries list: 
for country in countries:
    # Subset the points according to onshore/offshore
    points = europe_points.loc[europe_points.country==country].set_index(['grid_lat','grid_lon'])
    if onshore and not offshore:
        points = points.loc[~points.sea_node]
    elif not onshore and offshore:
        points = points.loc[points.sea_node]
    elif onshore and offshore:
        pass
    else:
        print('Either onshore or offshore must be set to TRUE')
    # Assign points to list
    points = list(points.index.unique())

    # Divide the points list into a set of bins depending on the bin_size given as a script argument. (Default=50)
    bins = int(np.ceil(len(points)/bin_size))
    for i in range(bins): # Loop thourgh points bins
        i+=1
        bin_string = f'_{i}-{bins}'

        # Check if the result already exists and skip the bin if it does
        file_path = os.path.join(results_path,country+bin_string+'.csv')
        if os.path.isfile(file_path):
            print(f'{file_path} already exists. Delete file to run this set.')
            continue
            
        # Generate the job submission string to be submitted to Euler
        bash_str = f'bsub -n {cores} -W {wall_time} -J "{country}-{i}" -oo {results_path}/lsf.{country}-{i}.txt '\
            f'python {SAF_directory}/scripts/optimization/plant_optimization.py '\
            f'--SAF_directory {SAF_directory} '\
            f'--results_path {results_path} '\
            f'--country {country} '\
            f'--year {year} '\
            f'--bin_number {i} '\
            f'--bin_size {bin_size} '\
            f'--MIPGap {MIPGap} '\
            f'--DisplayInterval {DisplayInterval} '\
            f'{offshore_flag} '\
            f'--save_operation '\
            f'--verbose '\
        # bash_str = f'python $HOME/EuroSAFs/scripts/03_plant_optimization/02_plant_optimization.py -d $HOME/EuroSAFs -c {country} -m {MIPGap} -i {DisplayInterval} -b {i} -n {bin_size} -v -s'
        # bash_str = f'python $HOME/GitHub/EuroSAFs/scripts/03_plant_optimization/02_plant_optimization.py -d $HOME/GitHub/EuroSAFs -c {country} -m {MIPGap} -i {DisplayInterval} -b {i} -n {bin_size} -v -s'
        
        # Execute the generated job submission bash command 
        os.system(bash_str)
        time.sleep(0.1)