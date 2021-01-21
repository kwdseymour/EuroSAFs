#!~/anaconda3/envs/GIS/bin/python
# coding: utf-8

import os
import pandas as pd
import sys
import argparse
import time

desc_text = 'This sets up the plant optimizer script.'
parser = argparse.ArgumentParser(description=desc_text)
parser.add_argument('-d','--SAF_directory',
    help='The path to the "SAFlogistics" directory',)
args = parser.parse_args()

SAF_directory = args.SAF_directory
results_path = os.path.join(SAF_directory,'results','02_plant_optimization')
if not os.path.isdir(results_path):
    os.mkdir(results_path)

eval_file_path = os.path.join(SAF_directory,'results','02_plant_optimization','eval_points.txt')

if os.path.isfile(eval_file_path):
    print('WARNING: The previous results have not been cleared.')
else:
    fp = open(eval_file_path, 'a')
    fp.write('Evaluation points:\n')