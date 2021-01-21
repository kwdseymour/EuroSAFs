#!~/anaconda3/envs/GIS/bin/python
# coding: utf-8

import os
import pandas as pd
import sys
import argparse
import time
import json

desc_text = 'This identifies missing evaluation points for each country and extracts them from other countries\' files if they are found. The results are saved to new files in the post_processed folder'
parser = argparse.ArgumentParser(description=desc_text)
parser.add_argument('-d','--SAF_directory',
    help='The path to the "SAFlogistics" directory',)
args = parser.parse_args()

SAF_directory = args.SAF_directory

results_path = os.path.join(SAF_directory,'results','02_plant_optimization')
eval_file_path = os.path.join(results_path,'eval_points.txt')
post_process_path = os.path.join(results_path,'post_processed')
if not os.path.isdir(post_process_path):
    os.mkdir(post_process_path)

with open(SAF_directory+'/data/Countries_WGS84/Europe_Evaluation_Points.json', 'r') as fp:
    europe_points_dict = json.load(fp)

countries = [x.replace('.csv','') for x in os.listdir(results_path)]
countries = [x for x in countries if x in europe_points_dict.keys()]
master_results = pd.concat([pd.read_csv(os.path.join(results_path,x+'.csv'),index_col=0) for x in countries])
master_results.set_index(['lat','lon'],inplace=True)
master_results = master_results.loc[~master_results.index.duplicated(keep='first')]

for country in countries:
    eval_points = [tuple(x) for x in europe_points_dict[country]]
    country_df = master_results.loc[master_results.index.isin(eval_points)]
    country_df = country_df.sort_index()
    country_df.to_csv(os.path.join(post_process_path,country+'.csv'))