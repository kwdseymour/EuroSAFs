#!~/anaconda3/envs/GIS/bin/python
# coding: utf-8

'''
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
'''

import os 
import numpy as np
import pandas as pd
idx = pd.IndexSlice
from gurobipy import * 
import matplotlib.pyplot as plt
import warnings
import logging
from .errors import *
from .utilities import create_logger

SAF_directory = os.path.dirname(__file__)
for i in range(3):
    SAF_directory = os.path.dirname(SAF_directory)

if 'cluster/home' in os.getcwd():
    # Uses the $SCRATCH environment variable to locate the scratch file if this module is run within Euler
    scratch_path = os.environ['SCRATCH']
else:
    scratch_path = os.path.join(SAF_directory,'scratch')

logger = create_logger(SAF_directory,__name__,__file__)

class Site():
    '''
    Contains wind and PV resource data for the given location within the provided country.
    
    The resources are extracted from PV and wind files specific to the given country.

    Instances of this class are intended to be attributed to "Plant" objects.
    '''
    def __init__(self,coordinates,country,PV_data_path=None,wind_data_path=None):
        self.country = country
        self.lat = coordinates[0]
        self.lon = coordinates[1]
        warnings.simplefilter("ignore", UserWarning)
        if PV_data_path == None:
            PV_data_path = os.path.join(SAF_directory,'data','PVGIS',country+'_PV.parquet.gzip')
        if wind_data_path == None:
            wind_data_path = os.path.join(SAF_directory,'results','01_merra_wind_preprocessing',country+'.parquet.gzip')
        PV_data = pd.read_parquet(PV_data_path)
        PV_data.sort_index(inplace=True)
        wind_data = pd.read_parquet(wind_data_path)
        wind_data.sort_index(inplace=True)
        warnings.simplefilter("default", UserWarning)
        
        if coordinates not in wind_data.index.droplevel(2).unique():
            raise CoordinateError(coordinates,'wind')
            logger.error(f'The given point ({coordinates}) was not found in the wind dataframe.')
        if coordinates not in PV_data.index.droplevel(2).unique():
            raise CoordinateError(coordinates,'PV')
            logger.error(f'The given point ({coordinates}) was not found in the PV dataframe.')
        self.PV_data =  PV_data.loc[idx[self.lat,self.lon]]
        self.wind_data =  wind_data.loc[idx[self.lat,self.lon]]
        assert len(self.PV_data)==len(self.wind_data)

class Component:
    '''
    Represents a component of a power plant and contains all the component specifications.

    An object of this class must be initialized with a DataFrame (specs argument) imported from the /data/plant_assumptions.xlsx file.
    The sheet contains cost and performance assumptions for all plant components.
    '''
    def __init__(self,name,specs,wind_class=None):
        self.specs_units={}
        component_specs = specs.loc[specs.index.str.contains(name+'_')].index
        for spec in component_specs:
            spec_name = spec.replace(name+'_','')
            if name=='wind':
                spec_name = spec_name.replace(wind_class+'_','')
            self.__setattr__(spec_name,specs.at[spec,'value'])
            self.specs_units[spec_name] = specs.at[spec,'units']
    
    def spec_units(self,spec):
        '''Returns the units of the given specification.'''
        return self.specs_units[spec]
            
class Plant:
    '''
    Holds specification data for various Power-to-Liquid plant components and is used as an input to the "plant_optimizer" function.
    The Power-to-Liquid plant specifications are extracted from /data/plant_assumptions.xlsx file to Component objects as attributes.
    The sheet contains cost and performance assumptions for all plant components.
    
    The plant can be initialized with a Site object containg location data and hourly PV and wind production data.
    In this case, the rated trubine power specification will be added.

    '''
    def __init__(self,site=None,specs_path=None):
        self.solved=False
        self.site=site
        if specs_path == None:
            specs_path = os.path.join(SAF_directory,'data','plant_assumptions.xlsx')
        specs = pd.read_excel(specs_path,sheet_name='data',index_col=0)
        component_names = ['wind','PV','battery','electrolyzer','CO2','H2stor','CO2stor','H2tL']
        for component_name in component_names:
            wind_spcap_class=None
            if component_name == 'wind' and not site == None:
                wind_spcap_class = site.wind_data['specific_capacity_class'][0]
            self.__setattr__(component_name,Component(component_name,specs,wind_spcap_class=wind_spcap_class))

        self.specs_units={}
        
        for i in specs.index:
            if all(x not in i for x in component_names):
                self.__setattr__(i,specs.at[i,'value'])
                self.specs_units[i] = specs.at[i,'units']
        if not self.site == None:
            self.site = site
            self.wind.__setattr__('rated_turbine_power',site.wind_data['rated_power_MW'][0]*1e3) # [kW]
            self.wind.__setattr__('turbine_type',site.wind_data['turbine_type'][0])
            self.wind.__setattr__('rotor_diameter',site.wind_data['rotor_diameter'][0]) # [m]
            self.wind.specs_units['rated_turbine_power'] = 'kW'
            self.wind.specs_units['rotor_diameter'] = 'm'
        assert self.kerosene_energy_fraction+self.naphtha_energy_fraction+self.diesel_energy_fraction == 1
        assert self.kerosene_mass_fraction+self.naphtha_mass_fraction+self.diesel_mass_fraction == 1
        assert self.kerosene_vol_fraction+self.naphtha_vol_fraction+self.diesel_vol_fraction == 1
        
    def spec_units(self,spec):
        '''Returns the units of the given specification.'''
        return self.specs_units[spec]
    
def get_country_points(country,wind_data_path=None):
    '''Returns all the MERRA points (0.5 x 0.625 degree geospatial resolution) that reside within the provided country's borders.'''
    if wind_data_path == None:
        wind_data_path = os.path.join(SAF_directory,'results','01_merra_wind_preprocessing',country+'.parquet.gzip')
    wind_data = pd.read_parquet(wind_data_path)
    points = list(wind_data.index.droplevel(2).unique())
    return points

def costs_NPV(capex,opex,discount_rate,lifetime,capacity):
    '''
    Returns the net present value of all costs associated with installation and operation of an asset with the given capacity.
    capex should be given in cost per unit for installed capacity (e.g. EUR/kW)
    opex should be given as a fraction of capex
    lifetime should be given in years
    the units of installed capacity must correspond to the capex units
    '''
    return np.sum([capex]+[capex*opex/(1+discount_rate)**n for n in np.arange(lifetime+1)]) * capacity

def optimize_plant(plant,threads=None,MIPGap=0.001,timelimit=1000,DisplayInterval=10,silent=True,print_status=False):
    '''Runs an optimizer to minimize the levelized cost of fuel production at the given coordinate location.
    - "plant" must be an instance of the Plant class.
    - "threads" sets the number of processor cores used by the optimizer. If this is set to "None", the default number of the optimizer is used.
    - "timelimit" is the timelimit of the Gurobi solver
    - "MIPGap" is the MIPGap of the Gurobi solver
    - "DisplayInterval" is the time interval of the Gurobi logger display if silent is set to False
    - "silent" can be set to False to print out all Gurobi console statements
    - "print_status" can be set to True to indicate whether the optimization was successful
    
    Sets the solved Gurobi Model to the Plant object (plant).
    '''
    
    logger.info(f'Beginning plant optimizer for point ({plant.site.lat},{plant.site.lon}) in {plant.site.country}...')
    if plant.site == None:
        raise Exception('A site is required for plant optimization but no site object has been assigned to this plant\'s "site" attribute.')
    
    time_vec = np.arange(0, plant.site.wind_data.index.nunique()) # hours of the year
    time_vec_ext = np.arange(0, plant.site.wind_data.index.nunique()+1) # for the final storage states
    assert len(time_vec) == 8784

    m = Model("Plant_Design")
    # clean the model
    m.remove(m.getVars())
    m.remove(m.getVars())
    if silent:
        m.setParam('OutputFlag', 0)

    # time limit, thread limit, and MIPGap
    m.params.timelimit = timelimit # seconds
    m.params.MIPGap = MIPGap
    m.params.DisplayInterval = DisplayInterval # time interval of the logger display
    if not threads == None:
        m.params.threads = threads # CPU cores available to optimizer

    # Define variables
    ## Units
    wind_units          = m.addVar(lb=plant.wind.min_units, ub=plant.wind.max_units, vtype=GRB.INTEGER, name="wind_units") # installed turbines
    PV_units            = m.addVar(lb=plant.PV.min_units, ub=plant.PV.max_units, vtype=GRB.INTEGER, name="PV_units") # installed 1 kW PV units
    electrolyzer_units  = m.addVar(lb=plant.electrolyzer.min_units, ub=plant.electrolyzer.max_units, vtype=GRB.INTEGER, name="electrolyzer_units") # installed 1 kW (electricity input) electrolyzer units
    CO2_units           = m.addVar(lb=plant.CO2.min_units, ub=plant.CO2.max_units, vtype=GRB.INTEGER, name="CO2_units") # installed 1 kg/hr (CO2 output) CO2 collector units
    H2tL_units          = m.addVar(lb=plant.H2tL.min_units, ub=plant.H2tL.max_units, vtype=GRB.INTEGER, name="H2tL_units") # installed 1 kW (jet fuel output) hydrogen-to-liquid units
    battery_units       = m.addVar(lb=plant.battery.min_units, ub=plant.battery.max_units, vtype=GRB.INTEGER, name="battery_units") # installed 1 kWh battery units
    H2stor_units        = m.addVar(lb=plant.H2stor.min_units, ub=plant.H2stor.max_units, vtype=GRB.INTEGER, name="H2stor_units") # installed 1 kWh hydrogen tank units
    CO2stor_units       = m.addVar(lb=plant.CO2stor.min_units, ub=plant.CO2stor.max_units, vtype=GRB.INTEGER, name="CO2stor_units") # installed 1 kg CO2 tank units

    ## Storage operation
    battery_chr_kWh    = m.addVars(time_vec, lb=0, ub=1e6, vtype=GRB.CONTINUOUS, name="battery_chr_kWh") # Battery charge at each hour
    battery_dis_kWh    = m.addVars(time_vec, lb=0, ub=1e6, vtype=GRB.CONTINUOUS, name="battery_dis_kWh") # Battery discharge  at each hour
    battery_state_kWh  = m.addVars(time_vec_ext, lb=0, ub=1e6, vtype=GRB.CONTINUOUS, name="battery_state_kWh") # Battery charge state at the beginning of each hour
    H2stor_chr_kWh     = m.addVars(time_vec, lb=0, ub=1e6, vtype=GRB.CONTINUOUS, name="H2stor_chr_kWh") # Hydrogen tank charge rate at each hour
    H2stor_dis_kWh     = m.addVars(time_vec, lb=0, ub=1e6, vtype=GRB.CONTINUOUS, name="H2stor_dis_kWh") # Hydrogen tank discharge rate at each hour
    H2stor_state_kWh   = m.addVars(time_vec_ext, lb=0, ub=1e6, vtype=GRB.CONTINUOUS, name="H2stor_state_kWh") # Hydrogen tank charge state at the beginning of each hour
    CO2stor_chr_kg     = m.addVars(time_vec, lb=0, ub=1e6, vtype=GRB.CONTINUOUS, name="CO2stor_chr_kg") # CO2 tank charge rate (kg/hr) at each hour
    CO2stor_dis_kg     = m.addVars(time_vec, lb=0, ub=1e6, vtype=GRB.CONTINUOUS, name="CO2stor_dis_kg") # CO2 tank discharge rate (kg/hr) at each hour
    CO2stor_state_kg   = m.addVars(time_vec_ext, lb=0, ub=1e6, vtype=GRB.CONTINUOUS, name="CO2stor_state_kg") # CO2 tank charge state (kg) at the beginning of each hour

    ## Curtailed electricity
    curtailed_el_kWh   = m.addVars(time_vec, lb=0, ub=100000, vtype=GRB.CONTINUOUS, name="curtailed_el_kWh") # Curtailed electricity production at each hour

    ## Electricity consumption for H2 production 
    H2_el_kWh          = m.addVars(time_vec, lb=0, ub=100000, vtype=GRB.CONTINUOUS, name="H2_el_kWh") # Electricity consumed for H2 production at each hour
    CO2_el_kWh         = m.addVars(time_vec, lb=0, ub=100000, vtype=GRB.CONTINUOUS, name="CO2_el_kWh") # Electricity consumed for CO2 production at each hour
    m.update()

    wind_production_kWh = plant.site.wind_data['kWh']*wind_units # [kWh]
    PV_production_kWh   = plant.site.PV_data['Wh']*PV_units/1e3 # [kWh]

    # el_production_kWh   = [wind_production_kWh[i] + PV_production_kWh[i] - curtailed_el_kWh[i]
    #                        + battery_dis_kWh[i] - battery_chr_kWh[i] for i in time_vec] # [kWh electricity produced per hour]
    H2_production_kWh   = [H2_el_kWh[i]*plant.electrolyzer.efficiency for i in time_vec] # [kWh hydrogen produced per hour]
    H2_consumption_kWh  = [H2_production_kWh[i] + H2stor_dis_kWh[i] - H2stor_chr_kWh[i] for i in time_vec] # [kWh hydrogen consumed per hour]
    CO2_production_kg   = [CO2_el_kWh[i]/plant.CO2.el_efficiency*1e3 for i in time_vec] # [kg CO2 produced per hour]
    CO2_consumption_kg  = [CO2_production_kg[i] + CO2stor_dis_kg[i] - CO2stor_chr_kg[i] for i in time_vec] # [tCO2 consumed per hour]
    fuel_production_kWh = [x*plant.H2tL.efficiency*plant.kerosene_energy_fraction for x in H2_consumption_kWh] # [kWh *jet* fuel per produced hour]


    # Define constraints
    # electricity use constraint
    for t in time_vec:
        m.addConstr(curtailed_el_kWh[t] <= wind_production_kWh[t] + PV_production_kWh[t])
        m.addConstr(battery_chr_kWh[t] <= wind_production_kWh[t] + PV_production_kWh[t] - curtailed_el_kWh[t])
        # m.addConstr(H2_el_kWh[t] <= el_production_kWh[t])
        m.addConstr(wind_production_kWh[t] + PV_production_kWh[t] + battery_dis_kWh[t] == curtailed_el_kWh[t] + battery_chr_kWh[t] + H2_el_kWh[t] + CO2_el_kWh[t])

    # storage constraints: 
    ## storage level constraint
    m.addConstrs(battery_state_kWh[i] <= battery_units*plant.battery.unit_capacity for i in time_vec_ext)
    m.addConstrs(H2stor_state_kWh[i] <= H2stor_units*plant.H2stor.unit_capacity for i in time_vec_ext)
    m.addConstrs(CO2stor_state_kg[i] <= CO2stor_units*plant.CO2.unit_capacity for i in time_vec_ext)
    ## initial/final storage level constraint
    m.addConstr(battery_state_kWh[0] == battery_state_kWh[time_vec_ext[-1]])
    m.addConstr(H2stor_state_kWh[0] == H2stor_state_kWh[time_vec_ext[-1]])
    m.addConstr(CO2stor_state_kg[0] == CO2stor_state_kg[time_vec_ext[-1]])
    ## storage operation constraint
    for t in time_vec:
        m.addConstr(H2stor_state_kWh[t+1] == H2stor_state_kWh[t] + H2stor_chr_kWh[t] - H2stor_dis_kWh[t])
        m.addConstr(CO2stor_state_kg[t+1] == CO2stor_state_kg[t] + CO2stor_chr_kg[t] - CO2stor_dis_kg[t])
        m.addConstr(battery_state_kWh[t+1] == battery_state_kWh[t] + battery_chr_kWh[t]*plant.battery.cycle_efficiency - battery_dis_kWh[t]/plant.battery.cycle_efficiency)
        m.addConstr(battery_chr_kWh[t] <= battery_units*plant.battery.unit_capacity*plant.battery.c_rate) # Charge rate constraints
        m.addConstr(battery_dis_kWh[t] <= battery_units*plant.battery.unit_capacity*plant.battery.c_rate) # Disharge rate constraints

    # electrolyzer operation input constraint
    for t in time_vec:
        m.addConstr(H2_el_kWh[t] <= electrolyzer_units*plant.electrolyzer.unit_capacity) # each electrolyzer_units can accept 1 kW of electricity input
        m.addConstr(H2_el_kWh[t] >= electrolyzer_units*plant.electrolyzer.unit_capacity*plant.electrolyzer.baseload)
    
    # CO2 capture operation input constraint
    for t in time_vec:
        m.addConstr(CO2_production_kg[t] <= CO2_units*plant.CO2.unit_capacity) # each CO2 capture unit can output 1 kg of CO2 per hour
        # Add a baseload constraint?

    # H2tL operation constraint
    for t in time_vec:
        m.addConstr(H2_consumption_kWh[t]*plant.H2tL.efficiency*plant.kerosene_energy_fraction <= H2tL_units*plant.H2tL.unit_capacity)
        m.addConstr(H2_consumption_kWh[t]*plant.H2tL.efficiency*plant.kerosene_energy_fraction >= H2tL_units*plant.H2tL.unit_capacity*plant.H2tL.baseload)
        m.addConstr(CO2_consumption_kg[t] == H2_consumption_kWh[t]*plant.H2tL.required_CO2) # Ratio of CO2 to H2 as input to process

    # fuel production constraint
    m.addConstr(quicksum(fuel_production_kWh)/1e6 >= plant.required_fuel)
    m.update()

    # Define objective function
    # ----- CHECK IF WIND CAPACITY IS IN RIGHT UNITS RELATIVE TO CAPEX!!! -------
    lifetime_wind_cost         = costs_NPV(capex=plant.wind.CAPEX,opex=plant.wind.OPEX,discount_rate=plant.discount_rate,lifetime=plant.lifetime,capacity=wind_units*plant.wind.rated_turbine_power) # Is capacity in the right units relative to CAPEX????
    lifetime_PV_cost           = costs_NPV(capex=plant.PV.CAPEX,opex=plant.PV.OPEX,discount_rate=plant.discount_rate,lifetime=plant.lifetime,capacity=PV_units*plant.PV.unit_capacity)
    lifetime_electrolyzer_cost = costs_NPV(capex=plant.electrolyzer.CAPEX,opex=plant.electrolyzer.OPEX,discount_rate=plant.discount_rate,lifetime=plant.lifetime,capacity=electrolyzer_units*plant.electrolyzer.unit_capacity)
    lifetime_CO2_cost          = costs_NPV(capex=plant.CO2.CAPEX,opex=plant.CO2.OPEX,discount_rate=plant.discount_rate,lifetime=plant.lifetime,capacity=CO2_units*plant.CO2.unit_capacity/1e3*8760) # Capacity converted to tons/year to match CAPEX units
    lifetime_battery_cost      = costs_NPV(capex=plant.battery.CAPEX,opex=plant.battery.OPEX,discount_rate=plant.discount_rate,lifetime=plant.lifetime,capacity=battery_units*plant.battery.unit_capacity)
    lifetime_H2stor_cost       = costs_NPV(capex=plant.H2stor.CAPEX,opex=plant.H2stor.OPEX,discount_rate=plant.discount_rate,lifetime=plant.lifetime,capacity=H2stor_units*plant.H2stor.unit_capacity)
    lifetime_CO2stor_cost      = costs_NPV(capex=plant.CO2stor.CAPEX,opex=plant.CO2stor.OPEX,discount_rate=plant.discount_rate,lifetime=plant.lifetime,capacity=CO2stor_units*plant.CO2stor.unit_capacity/1e3) # Capacity converted to tons to match CAPEX units
    lifetime_H2tL_cost         = costs_NPV(capex=plant.H2tL.CAPEX,opex=plant.H2tL.OPEX,discount_rate=plant.discount_rate,lifetime=plant.lifetime,capacity=H2tL_units*plant.H2tL.unit_capacity)
    lifetime_cost              = lifetime_wind_cost + lifetime_PV_cost + lifetime_electrolyzer_cost + lifetime_CO2_cost + lifetime_battery_cost + lifetime_H2stor_cost + lifetime_CO2stor_cost + lifetime_H2tL_cost # EUR

    # Set objective function
    m.setObjective(lifetime_cost,sense=GRB.MINIMIZE)
    m.update()

    # Call the optimizer
    m.update()
    m.optimize()
    
    logger.info(f'Plant optimizer for point ({plant.site.lat},{plant.site.lon}) in {plant.site.country} finished after {m.Runtime:.0f} seconds.')
    if m.status == 2:
        final_message = "Optimal Solution Found. Congratulations."
        logger.info(f'{final_message}')
    elif m.status == 9: 
        final_message = "Solver took longer than the timelimit permitted."
        logger.error(f'{final_message}')
    else: 
        final_message = "Something went wrong."
        logger.error(f'{final_message}')
        raise OptimizerError((plant.site.lat,plant.site.lon))
    if print_status:
        print(final_message)
    
    plant.m = m

def unpack_design_solution(plant,unpack_operation=False):
    '''Parses the results of the optimized plant and save the values to attributes of the given plant object.
    Set unpack_operation to True in order to unpack the hourly operation of applicable components (e.g. wind, PV, CO2, battery)
    '''
    plant.wind_units         = plant.m.getVarByName('wind_units').x
    plant.PV_units           = plant.m.getVarByName('PV_units').x
    plant.electrolyzer_units = plant.m.getVarByName('electrolyzer_units').x
    plant.CO2_units          = plant.m.getVarByName('CO2_units').x
    plant.battery_units      = plant.m.getVarByName('battery_units').x
    plant.H2stor_units       = plant.m.getVarByName('H2stor_units').x
    plant.CO2stor_units      = plant.m.getVarByName('CO2stor_units').x
    plant.H2tL_units         = plant.m.getVarByName('H2tL_units').x
    plant.NPV                = plant.m.ObjVal
    CAPEXes = {}
    for component in ['PV','battery','electrolyzer','H2stor','H2tL']:
        CAPEXes[component] = plant.__getattribute__(component+'_units') * plant.__getattribute__(component).unit_capacity * plant.__getattribute__(component).CAPEX
    CAPEXes['wind'] = plant.wind.rated_turbine_power * plant.wind_units * plant.wind.CAPEX
    CAPEXes['CO2'] = plant.CO2.unit_capacity/1e3*8760 * plant.CO2_units * plant.CO2.CAPEX
    CAPEXes['CO2'] = plant.CO2stor.unit_capacity/1e3 * plant.CO2stor_units * plant.CO2stor.CAPEX
    plant.CAPEXes = CAPEXes
    plant.CAPEX              = np.sum(list(CAPEXes.values()))
    plant.LCOF_MWh           = plant.NPV/(plant.required_fuel*1e3*plant.lifetime)
    plant.LCOF_liter         = plant.LCOF_MWh/3.6e9*plant.kerosene_LHV*0.8
    if unpack_operation:
        unpack_operation_solution(plant)
    plant.solved=True

def unpack_operation_solution(plant):
    '''Parses the hourly operation results of the optimized plant and save the values to attributes of the given plant object.
    
    The results are also saved to the attribute named "operation" as a DatFrame for easy viewing.
    '''
    plant.wind_production_MWh = plant.site.wind_data['kWh']*plant.m.getVarByName('wind_units').x/1e3 # [MWh]
    plant.PV_production_MWh = plant.site.PV_data['Wh']*plant.m.getVarByName('PV_units').x/1e6 # [MWh]
    
    time_vec = np.arange(0, plant.site.wind_data.index.nunique()) # hours of the year
    time_vec_ext = np.arange(0, plant.site.wind_data.index.nunique()+1) # for the final storage states
    

    plant.battery_chr_MWh   = []
    plant.battery_dis_MWh   = []
    plant.battery_state_MWh = []
    plant.H2stor_chr_MWh    = []
    plant.H2stor_dis_MWh    = []
    plant.H2stor_state_MWh  = []
    plant.CO2stor_chr_ton    = []
    plant.CO2stor_dis_ton    = []
    plant.CO2stor_state_ton  = []
    plant.curtailed_el_MWh  = []
    plant.H2_el_MWh         = []
    plant.CO2_el_MWh         = []
    for entry in plant.m.getVars():
        var_name = entry.varname
        var_value = entry.x
        if 'battery_chr_kWh' in var_name:
            plant.battery_chr_MWh.append(var_value/1e3)
        elif 'battery_dis_kWh' in var_name:
            plant.battery_dis_MWh.append(var_value/1e3)
        elif 'battery_state_kWh' in var_name:
            plant.battery_state_MWh.append(var_value/1e3)
        elif 'curtailed_el_kWh' in var_name:
            plant.curtailed_el_MWh.append(var_value/1e3)
        elif 'H2_el_kWh' in var_name:
            plant.H2_el_MWh.append(var_value/1e3)
        elif 'CO2_el_kWh' in var_name:
            plant.CO2_el_MWh.append(var_value/1e3)
        elif 'H2stor_chr_kWh' in var_name:
            plant.H2stor_chr_MWh.append(var_value/1e3)
        elif 'H2stor_dis_kWh' in var_name:
            plant.H2stor_dis_MWh.append(var_value/1e3)
        elif 'H2stor_state_kWh' in var_name:
            plant.H2stor_state_MWh.append(var_value/1e3)
        elif 'CO2stor_chr_kg' in var_name:
            plant.CO2stor_chr_ton.append(var_value/1e3)
        elif 'CO2stor_dis_kg' in var_name:
            plant.CO2stor_dis_ton.append(var_value/1e3)
        elif 'CO2stor_state_kg' in var_name:
            plant.CO2stor_state_ton.append(var_value/1e3)
            
    plant.battery_flow_MWh = np.subtract(plant.battery_chr_MWh,plant.battery_dis_MWh)
    plant.H2stor_flow_MWh = np.subtract(plant.H2stor_chr_MWh,plant.H2stor_dis_MWh)
    plant.CO2stor_flow_ton = np.subtract(plant.CO2stor_chr_ton,plant.CO2stor_dis_ton)

    # plant.el_production_MWh = [plant.wind_production_MWh[i] + plant.PV_production_MWh[i] - plant.curtailed_el_MWh[i]
    #                        + plant.battery_dis_MWh[i] - plant.battery_chr_MWh[i] for i in time_vec] # [MWh electricity produced per hour]

    plant.H2_production_MWh   = [plant.H2_el_MWh[i]*plant.electrolyzer.efficiency for i in time_vec] # [MWh hydrogen produced per hour]
    plant.H2_consumption_MWh  = [plant.H2_production_MWh[i] + plant.H2stor_dis_MWh[i] - plant.H2stor_chr_MWh[i] for i in time_vec] # [MWh hydrogen consumed per hour]
    plant.CO2_production_ton   = [plant.CO2_el_MWh[i]*1e3/plant.CO2.el_efficiency for i in time_vec] # [tCO2 produced per hour]
    plant.CO2_consumption_ton  = [plant.CO2_production_ton[i] + plant.CO2stor_dis_ton[i] - plant.CO2stor_chr_ton[i] for i in time_vec] # [tCO2 consumed per hour]
    plant.fuel_production_MWh = [x*plant.H2tL.efficiency*plant.kerosene_energy_fraction for x in plant.H2_consumption_MWh] # [MWh *jet* fuel per produced hour]
    
    plant.operation = pd.DataFrame({
                                    'wind_production_MWh' : list(plant.wind_production_MWh),
                                    'PV_production_MWh' : list(plant.PV_production_MWh),
                                    'curtailed_el_MWh' : plant.curtailed_el_MWh,
                                    'battery_flow_MWh' : plant.battery_flow_MWh,
                                    'battery_chr_MWh' : plant.battery_chr_MWh,
                                    'battery_dis_MWh' : plant.battery_dis_MWh,
                                    'battery_state_MWh' : plant.battery_state_MWh[:-1],
                                    # 'el_production_MWh' : plant.el_production_MWh,
                                    'H2_el_MWh' : plant.H2_el_MWh,
                                    'CO2_el_MWh' : plant.CO2_el_MWh,
                                    'H2_production_kWh' : plant.H2_production_MWh,
                                    'H2stor_flow_MWh' : plant.H2stor_flow_MWh,
                                    'H2stor_chr_MWh' : plant.H2stor_chr_MWh,
                                    'H2stor_dis_MWh' : plant.H2stor_dis_MWh,
                                    'H2stor_state_MWh' : plant.H2stor_state_MWh[:-1],
                                    'CO2_consumption_kg' : plant.CO2_consumption_ton,
                                    'CO2_production_ton' : plant.CO2_production_ton,
                                    'CO2stor_flow_ton' : plant.CO2stor_flow_ton,
                                    'CO2stor_chr_ton' : plant.CO2stor_chr_ton,
                                    'CO2stor_dis_ton' : plant.CO2stor_dis_ton,
                                    'CO2stor_state_ton' : plant.CO2stor_state_ton[:-1],
                                    'CO2_consumption_kg' : plant.CO2_consumption_ton,
                                    'fuel_production_MWh' : plant.fuel_production_MWh,
                                   },index=plant.site.wind_data.index)

