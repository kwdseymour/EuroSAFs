#!~/anaconda3/envs/GIS/bin/python
# coding: utf-8

import os 
import numpy as np
import pandas as pd
idx = pd.IndexSlice
import matplotlib.pyplot as plt

def print_solution_summary(plant):
    '''Prints a summary the unpacked plant optimization results'''
    if not plant.solved:
        raise Exception('Plant has no solution. Use "unpack_solution" to assign a solution to the plant first.')
    print('Installed wind capacity -- {} MW ({:.0f} turbines).'.format(plant.wind_units*plant.wind.rated_turbine_power/1e3,plant.wind_units))
    print('Installed PV capacity -- {} MW.'.format(plant.PV_units*plant.PV.unit_capacity/1e3))
    print('Installed electrolyzer capacity -- {} MW electric.'.format(plant.electrolyzer_units*plant.electrolyzer.unit_capacity/1e3))
    print('Installed CO2 capture capacity -- {} kg/hr.'.format(plant.CO2_units*plant.CO2.unit_capacity))
    print('Installed hydrogen-to-liquid component capacity -- {} MW.'.format(plant.H2tL_units*plant.H2tL.unit_capacity/1e3))
    print('Installed battery capacity -- {} MWh.'.format(plant.battery_units*plant.battery.unit_capacity/1e3))
    print('Installed hydrogen storage tank capacity -- {} MWh.'.format(plant.H2stor_units*plant.H2stor.unit_capacity/1e3))
    print('Installed CO2 storage tank capacity -- {} tons.'.format(plant.CO2stor_units*plant.CO2stor.unit_capacity/1e3))
    try:
        print(f'Total curtailed electricity (yearly): {np.sum(plant.curtailed_el_MWh)/1e3} GWh')
    except:
        pass
    print('-----\nNet present value of plant -- {:.1f} million EUR.'.format(plant.NPV/1e6))
    print('CAPEX of plant -- {:.1f} million EUR.'.format(plant.CAPEX/1e6))
    print('Levelized cost of fuel -- {:.2f} EUR/MWh ({:.2f} EUR/liter).'.format(plant.LCOF_MWh,plant.LCOF_liter))
    
def plot_plant_operation(plant,xlim=None,fuel_rate=True,figsize=(20,14)):
    '''Creates multiple plots showing the operation of the optimized plant. Use xlim to set the boundaries of the x axis and zoom into areas of  interest in the plots.'''
    nrows = 5 if fuel_rate else 4
    fig,axes = plt.subplots(nrows=nrows,figsize=figsize)
    axes.reshape(nrows,1)

    time_vec = np.arange(0, plant.site.wind_data.index.nunique()) # hours of the year
    time_vec_ext = np.arange(0, plant.site.wind_data.index.nunique()+1) # for the final storage states
    
    # Electricity production
    ## Wind
    axes[0].plot(time_vec,plant.wind_production_MWh,label='Wind [MW]')
    ## Solar
    axes[0].plot(time_vec,plant.PV_production_MWh,label='PV [MW]',color='orange')
    ## Curtailed
    axes[0].plot(time_vec,plant.curtailed_el_MWh,color='brown',label='Curtailed [MW]')

    axes[0].set_ylim(0,max(max(plant.wind_production_MWh),max(plant.PV_production_MWh),max(plant.curtailed_el_MWh))*1.2)
    axes[0].legend()
    # axes3.legend(loc='upper right')

    # Battery 
    ## Battery charge/discharge
    axes[1].plot(time_vec,plant.battery_flow_MWh,color='purple',label='Battery charge/discharge rate [MW]')
    axes[1].axhline(color='grey',linestyle='--')
    ## Battery state
    axes2 = plt.twinx(axes[1])
    axes2.plot(time_vec_ext,plant.battery_state_MWh,color='green',label='Battery state [MWh]')

    axes[1].set_ylim(min(plant.battery_flow_MWh),max(plant.battery_flow_MWh)*1.3)
    axes2.set_ylim(min(plant.battery_flow_MWh)/max(plant.battery_flow_MWh)*max(plant.battery_state_MWh),max(plant.battery_state_MWh)*1.3)
    axes[1].legend(loc='upper left')
    axes2.legend(loc='upper right')

    # Storage tanks 
    ## Hydrogen tank state
    axes[2].plot(time_vec_ext,plant.H2stor_state_MWh,color='cyan',label='Hydrogen tank state [MWh]')
    ## CO2 tank state
    axes3 = plt.twinx(axes[2])
    axes3.plot(time_vec_ext,plant.CO2stor_state_ton,color='brown',label='CO2 tank state [tCO2]')

    # axes[2].set_ylim(min(plant.H2stor_flow_MWh)/max(plant.H2stor_flow_MWh)*max(plant.H2stor_state_MWh),max(plant.H2stor_state_MWh)*1.3)
    # axes3.set_ylim(min(plant.CO2stor_flow_ton)/max(plant.CO2stor_flow_ton)*max(plant.CO2stor_state_ton),max(plant.CO2stor_state_ton)*1.3)
    # axes[2].set_ylim(min(plant.H2stor_flow_MWh)/max(plant.H2stor_flow_MWh)*max(plant.H2stor_state_MWh),max(plant.H2stor_state_MWh)*1.3)
    # axes3.set_ylim(min(plant.CO2stor_flow_ton)/max(plant.CO2stor_flow_ton)*max(plant.CO2stor_state_ton),max(plant.CO2stor_state_ton)*1.3)
    axes[2].legend(loc='upper left')
    axes3.legend(loc='upper right')

    # Electricity flow
    ## Hydrogen production
    electricity_generation = plant.wind_production_MWh.reset_index(drop=True)+plant.PV_production_MWh.reset_index(drop=True)-plant.curtailed_el_MWh
    axes[3].plot(time_vec,electricity_generation,color='green',label='Electricity generation (wind + PV) [MW]')
    axes[3].plot(time_vec,plant.battery_flow_MWh,color='purple',alpha=0.8,label='Battery charge/discharge rate [MW]')
    axes[3].plot(time_vec,plant.H2_el_MWh,color='cyan',alpha=0.8,label='Electricity to electrolyzer [MW]')
    axes[3].plot(time_vec,plant.CO2_el_MWh,color='brown',alpha=0.8,label='Electricity to CO2 capture [MW]')
    ## CO2 production
    # axes3 = plt.twinx(axes[3])
    # axes3.plot(time_vec,plant.CO2_production_ton,color='brown',label='CO2 production rate [tCO2/hr]')

    axes[3].set_ylim(min(plant.battery_flow_MWh),max(max(electricity_generation),max(plant.H2_el_MWh),max(plant.battery_flow_MWh),max(plant.CO2_el_MWh))*1.3)
    # axes3.set_ylim(min(plant.CO2_production_ton)/max(plant.CO2_production_ton)*max(plant.CO2_production_ton),max(plant.CO2_production_ton)*1.3)
    axes[3].legend(loc='upper left')
    # axes3.legend(loc='upper right')

    if fuel_rate:
        # Fuel production
        # axes[4].plot(time_vec,plant.H2_production_MWh,color='gray',label='Hydrogen production rate [MW]')
        axes[4].plot(time_vec,plant.fuel_production_MWh,color='k',label='Jet fuel production rate [MW]')
        axes[4].legend()
        axes[4].set_ylim(0,max(max(plant.fuel_production_MWh),max(plant.H2_production_MWh))*1.1)
        axes[4].set_xlabel('Hours')

    if xlim==None:
        xlim = (0,len(time_vec))
    for ax in axes:
        ax.set_xlim(xlim)
    plt.tight_layout()

# def plot_plant_operation(plant,xlim=None,figsize=(20,14)):
#     fig,axes = plt.subplots(nrows=5,figsize=figsize)
#     axes.reshape(5,1)

#     time_vec = np.arange(0, plant.site.wind_data.index.nunique()) # hours of the year
#     time_vec_ext = np.arange(0, plant.site.wind_data.index.nunique()+1) # for the final storage states
    
#     # Electricity production
#     ## Wind
#     axes[0].plot(time_vec,plant.wind_production_MWh,label='Wind [MW]')
#     ## Solar
#     axes[0].plot(time_vec,plant.PV_production_MWh,label='PV [MW]',color='orange')
#     ## Curtailed
#     axes[0].plot(time_vec,plant.curtailed_el_MWh,color='brown',label='Curtailed [MW]')

#     axes[0].set_ylim(0,max(max(plant.wind_production_MWh),max(plant.PV_production_MWh),max(plant.curtailed_el_MWh))*1.2)
#     axes[0].legend()
#     # axes3.legend(loc='upper right')

#     # Battery 
#     ## Battery charge/discharge
#     axes[1].plot(time_vec,plant.battery_flow_MWh,color='purple',label='Battery charge/discharge rate [MW]')
#     axes[1].axhline(color='grey',linestyle='--')
#     ## Battery state
#     axes2 = plt.twinx(axes[1])
#     axes2.plot(time_vec_ext,plant.battery_state_MWh,color='green',label='Battery state [MWh]')

#     axes[1].set_ylim(min(plant.battery_flow_MWh),max(plant.battery_flow_MWh)*1.3)
#     axes2.set_ylim(min(plant.battery_flow_MWh)/max(plant.battery_flow_MWh)*max(plant.battery_state_MWh),max(plant.battery_state_MWh)*1.3)
#     axes[1].legend(loc='upper left')
#     axes2.legend(loc='upper right')

#     # Hydrogen tank 
#     ## Hydrogen tank charge/discharge
#     axes[2].plot(time_vec,plant.H2stor_flow_MWh,color='purple',label='Hydrogen tank charge/discharge rate [MW]')
#     axes[2].axhline(color='grey',linestyle='--')
#     ## Hydrogen tank state
#     axes3 = plt.twinx(axes[2])
#     axes3.plot(time_vec_ext,plant.H2stor_state_MWh,color='green',label='Hydrogen tank state [MWh]')

#     axes[2].set_ylim(min(plant.H2stor_flow_MWh),max(plant.H2stor_flow_MWh)*1.3)
#     axes3.set_ylim(min(plant.H2stor_flow_MWh)/max(plant.H2stor_flow_MWh)*max(plant.H2stor_state_MWh),max(plant.H2stor_state_MWh)*1.3)
#     axes[2].legend(loc='upper left')
#     axes3.legend(loc='upper right')

#     # CO2 tank 
#     ## CO2 tank charge/discharge
#     axes[3].plot(time_vec,plant.CO2stor_flow_ton,color='purple',label='CO2 tank charge/discharge rate [tCO2/hr]')
#     axes[3].axhline(color='grey',linestyle='--')
#     ## CO2 tank state
#     axes3 = plt.twinx(axes[3])
#     axes3.plot(time_vec_ext,plant.CO2stor_state_ton,color='green',label='CO2 tank state [tCO2]')

#     axes[3].set_ylim(min(plant.CO2stor_flow_ton),max(plant.CO2stor_flow_ton)*1.3)
#     axes3.set_ylim(min(plant.CO2stor_flow_ton)/max(plant.CO2stor_flow_ton)*max(plant.CO2stor_state_ton),max(plant.CO2stor_state_ton)*1.3)
#     axes[3].legend(loc='upper left')
#     axes3.legend(loc='upper right')

#     # Fuel production
#     axes[4].plot(time_vec,plant.H2_production_MWh,color='gray',label='Hydrogen production rate [MW]')
#     axes[4].plot(time_vec,plant.fuel_production_MWh,color='k',label='Jet fuel production rate [MW]')
#     axes[4].legend()
#     axes[4].set_ylim(0,max(max(plant.fuel_production_MWh),max(plant.H2_production_MWh))*1.1)
#     axes[4].set_xlabel('Hours')

#     if xlim==None:
#         xlim = (0,len(time_vec))
#     for ax in axes:
#         ax.set_xlim(xlim)
#     plt.tight_layout()