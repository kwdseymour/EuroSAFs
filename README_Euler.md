# Description

This describes the steps necessary to configure EuroSAFs to run on an Euler account.

# Git Repository Cloning
Clone the git repository in the $HOME directory if possible (this decreases the likelihood of file import errors within some scripts):\
`git clone https://github.com/kwdseymour/EuroSAFs.git`

# Initial Euler Environment Set up 
Load a Python module:\
`module load gcc/4.8.2 python/3.7.1`

Create a Python virtual environment:\
`python -m venv --system-site-packages EuroSAFs`\
The option --system-site-packages includes Python packages from host Python also in the virtual environment. This is needed to access Gurobi.

Activate the virtual environment:\
`source EuroSAFs/bin/activate`

Install packages in the environment using the requirements.txt file:\
`python -m pip install -r EuroSAFs/requirements.txt`\
(ammend path to requirements.txt file as needed)

# Subsequent Environment Activations (Optional)
To minimize the number of steps required to configure the environments and update files in future runs, it can be helpful to create a shortcut (alias).

Open .bash_profile in your $HOME directory and paste the following line at the bottom:\
`alias esConfig='module load gcc/4.8.2 python/3.7.1; source EuroSAFs/bin/activate; cd $HOME/EuroSAFs; git pull'`\
(ammend path to the cloned EuroSAFs directory as needed)

By running this new alias, `esConfig`, after each subsequent connection to Euler, the environment will be set up as needed to run scripts.

# Results Directory Setup
The SAF plant simulations require wind and solar PV production data generated in a previous stage of research. These data need to be located in a "results" folder located in the parent EuroSAFs directory, which is also where the results of the plant simulations will be saved. Please contact a contributor to obtain access to the necessary files. Once you have done so, configure the wind & PV data folders with the following paths:
- EuroSAFs/results/PV_power_output
- EuorSAFs/results/wind_power_output
Each directory should contain a parquet file for each country to be simulated.

# Running Simulations
## run_plant_optimization.py
This script submits jobs to run the plant optimizer for all European countries. The number of jobs submitted when running this script is on the order of 100. Call the script from the EuroSAFs directory like so:\
`python scripts/optimization/run_plant_optimization.py -y 2020`
