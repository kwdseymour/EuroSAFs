# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import os
import json
import sys

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Output, Input

import numpy as np
import pandas as pd
import geopandas as gpd
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(1,'../optimization/')
from plant_optimization.notebook_tools import *

# %run style_sheet_MH.ipynb
plt.style.reload_library()
plt.style.use('EuroSAFs_pub')

europe_grid = gpd.read_file('../../data/Countries_WGS84/processed/Europe_Evaluation_Grid.shp')
europe_grid.rename(columns={'grid_lat':'lat','grid_lon':'lon'},inplace=True)
europe_grid['sea_node'] = europe_grid.sea_node.astype(bool)
europe_borders = gpd.read_file('../../data/Countries_WGS84/processed/Europe_WGS84.shp')

EU_EFTA = europe_grid.country.unique()

results_path = os.path.join('../../results/plant_optimization/final_results') # Points to the location of the results csv files

specs = pd.read_excel('../../data/plant_assumptions.xlsx',sheet_name='data',index_col=0)
specs_names = specs.index

# Override wind turbine spacing assumption. Fix in plant assumptions spreadsheet 
# (https://windharvest.com/wp-content/uploads/2017/03/Optimal-Turbine-Spacing-In-Fully-Developed-Wind-Farm-Boundary-Layers-Johan-Meyers-and-Charles-Meneveau-Wiley-Online-Library-2011.pdf)
specs.rename(index={'wind_turbine_spacing':'wind_turbine_spacing_onshore'}, inplace=True)
specs = specs.append(pd.DataFrame({'value_2020':15},index=['wind_turbine_spacing_offshore']))

lc_path = '../../results/land_availability/00_land_availability_extended'
sea_area_path = '../../results/land_availability/sea_area'
land_data = pd.DataFrame()
sea_data = pd.DataFrame()
for country in EU_EFTA:
    country_land_data = pd.read_csv(f'{lc_path}/{country}.csv')
    country_land_data['country'] = country

    land_data = land_data.append(country_land_data)
    try:
        sea_data = sea_data.append(pd.read_csv(f'{sea_area_path}/{country}.csv'))
    except FileNotFoundError:
        continue
land_data['sea_node'] = False
sea_data['sea_node'] = True
# Correct an error in which the Norway points are given NaN for the country name
land_data.loc[land_data.country.isna(),'country'] = 'Norway'
sea_data.loc[sea_data.country.isna(),'country'] = 'Norway'
lc_data = pd.concat([land_data,sea_data]).sort_values(['country','lat','lon']).reset_index(drop=True)
# lc_data.crs = gpd.GeoDataFrame(lc_data,crs=europe_grid.crs)

corine_code = pd.read_excel('../../data/CORINE_legend.xlsx',usecols=[0,7])
corine_code = corine_code.rename(columns={'GRID_CODE':'code','LABEL3':'name'}).dropna()

# Review the assumption of PV area!!! https://www.nrel.gov/docs/fy13osti/56290.pdf
pv_area_acres = 5.5e-3 # acres/kW
acres_per_sqkm = 247.105 
pv_area_sqkm = acres_per_sqkm/pv_area_acres #kW/sqkm

results = {}
for year in [2020,2030,2040,2050]:
    df = pd.read_csv(os.path.join(results_path,f'{year}.csv'))
    df.drop(columns=['geometry'],inplace=True)
    gdf = gpd.GeoDataFrame(df.merge(europe_grid,on=['lat','lon','country','sea_node'],how='left'))

    spacing = gdf['sea_node'].apply(lambda x: {True:specs.at['wind_turbine_spacing_offshore',f'value_2020'],False:specs.at['wind_turbine_spacing_onshore',f'value_2020']}[x])
    gdf['turbine_area_sqkm'] = (gdf.rotor_diameter*spacing)**2*df.wind_turbines/1e6
#     gdf['pv_area_sqkm'] = gdf.PV_capacity_MW*1e3/specs.at['PV_peak_per_area',f'value_2020']/1e6
    gdf['pv_area_sqkm'] = gdf.PV_capacity_MW*1e3/pv_area_sqkm
    gdf['plant_area_sqkm'] = gdf['turbine_area_sqkm'] + gdf['pv_area_sqkm']

    results[year] = gdf


def calculate_production_potential(gdf,land_cover_types):
    land_cover_types = [str(int(x)) for x in land_cover_types]

    combined = gdf.merge(lc_data,on=['lat','lon','country','sea_node'],how='left')
    combined['avail_area_sqkm'] = combined['avail_area_sqkm'].where(~combined['avail_area_sqkm'].isna(),combined[land_cover_types].sum(axis=1))
    combined = gpd.GeoDataFrame(combined, crs=gdf.crs)

    combined['plants'] = combined.avail_area_sqkm/combined.plant_area_sqkm
    combined['production_GWh'] = combined.plants * specs.at['required_fuel',f'value_2020']
    combined['production_liters'] = combined.production_GWh*3.6e12/specs.at['kerosene_LHV',f'value_2020']/0.8
    return combined

# IATA average jet fuel price for 2021: $77.6/bbl (https://www.iata.org/en/publications/economics/fuel-monitor/)
# Average exchange rate 2021: 1.183 USD = 1 EUR (https://www.exchangerates.org.uk/EUR-USD-spot-exchange-rates-history-2021.html)
# IATA average jet fuel price for 2019: $79.6/bbl (https://iata.org.xy2401.com/publications/economics/fuel-monitor/Pages/index.aspx.html)
# Average exchange rate 2019: 1.1199 USD = 1 EUR (https://www.exchangerates.org.uk/EUR-USD-spot-exchange-rates-history-2019.html)
# 1 US bbl oil = 158.99 L

exchange_rate_2019 = 1.1199 #USD to EUR
exchange_rate_2021 = 1.18 #USD to EUR
exchange_rate_2022 = 1.13 #USD to EUR
bbl_to_liter = 158.99 #US bbl oil to liters
fossil_price_2019 = 79.6/exchange_rate_2019/bbl_to_liter
fossil_price_2021 = 77.6/exchange_rate_2021/bbl_to_liter

EU32_fuel_demand_kg = 60e9 # 60 Mt jet fuel
kerosene_density = .820 #kg/l
EU32_fuel_demand_liters = EU32_fuel_demand_kg/kerosene_density

def calculate_cost_curve(df,max_lcof = 5):
    cost_curve = df.loc[df.LCOF_liter<=max_lcof].sort_values('LCOF_liter').reset_index()
    # cost_curve = cost_curve.loc[cost_curve.plants<=100].reset_index()
    # cost_curve = df.sort_values('LCOF_liter').reset_index()
    # REMOVE NODES THAT HAVE NO WIND GENERATION
#     no_wind_mask = cost_curve.wind_capacity_MW <= 0
#     print(f'{no_wind_mask.sum()} locations without wind turbines were given override production volumes {year}.')
#     cost_curve.loc[no_wind_mask,'production_liters'] = cost_curve.loc[~no_wind_mask,'production_liters'].max()
    cost_curve['production_liters_cumsum'] = cost_curve.production_liters.cumsum()
    return cost_curve


app = Dash()
app.layout = html.Div([
    html.Div([
        dcc.Checklist(
            id='land_sea1',
            options=[{'label': 'Onshore Nodes', 'value': 'onshore'},{'label': 'Offshore Nodes', 'value': 'offshore'}],
            value=['onshore','offshore']),
        dcc.Checklist(
            id='years1',
            options=[{'label': x, 'value': x} for x in results.keys()],
            value=[2020,2030,2040,2050]),
        dcc.Dropdown(
            id='land_cover_types1',
            options=[{'label': f"{row['code']:.0f}. {row['name']}", 'value': int(row['code'])} for i,row in corine_code.iterrows()],
            multi=True,
            value=[32]),
        dcc.Graph(id='cost_curve1')],
        style={'width':'33%', 'display':'inline-block'}),
    html.Div([
        dcc.Checklist(
            id='land_sea2',
            options=[{'label': 'Onshore Nodes', 'value': 'onshore'},{'label': 'Offshore Nodes', 'value': 'offshore'}],
            value=['onshore','offshore']),
        dcc.Checklist(
            id='years2',
            options=[{'label': x, 'value': x} for x in results.keys()],
            value=[2020,2030,2040,2050]),
        dcc.Dropdown(
            id='land_cover_types2',
            options=[{'label': f"{row['code']:.0f}. {row['name']}", 'value': int(row['code'])} for i,row in corine_code.iterrows()],
            multi=True,
            value=[18,28,29,32]),
        dcc.Graph(id='cost_curve2')],
        style={'width':'33%', 'display':'inline-block'}),
    html.Div([
        dcc.Checklist(
            id='land_sea3',
            options=[{'label': 'Onshore Nodes', 'value': 'onshore'},{'label': 'Offshore Nodes', 'value': 'offshore'}],
            value=['onshore']),
        dcc.Checklist(
            id='years3',
            options=[{'label': x, 'value': x} for x in results.keys()],
            value=[2020,2030,2040,2050]),
        dcc.Dropdown(
            id='land_cover_types3',
            options=[{'label': f"{row['code']:.0f}. {row['name']}", 'value': int(row['code'])} for i,row in corine_code.iterrows()],
            multi=True,
            value=[29,32]),
        dcc.Graph(id='cost_curve3')],
        style={'width':'33%', 'display':'inline-block'})
])

@app.callback(Output('cost_curve1', 'figure'),
              [Input('land_sea1', 'value'),
               Input('years1', 'value'),
               Input('land_cover_types1', 'value')])
def update_graph1(land_sea, years, land_cover_types):
    fig = make_subplots(rows=1, cols=1, horizontal_spacing=0.1,subplot_titles=['SAF cost curve'])
    for year in years:
        production_potential = calculate_production_potential(results[year],land_cover_types)
        if 'onshore' not in land_sea:
            production_potential = production_potential.loc[production_potential.sea_node]
        if 'offshore' not in land_sea:
            production_potential = production_potential.loc[~production_potential.sea_node]
        cost_curve = calculate_cost_curve(production_potential)

        # fig.append_trace({'x':cost_curve.production_liters_cumsum/1e9,'y':cost_curve.LCOF_liter,'name':'Cost','mode':'lines','type':'scatter','line':{'color':"#4285F4"}},1,1)
        fig.append_trace({'x':cost_curve.production_liters_cumsum/1e9,'y':cost_curve.LCOF_liter,'name':year,'mode':'lines','type':'scatter'},1,1)
    return fig

@app.callback(Output('cost_curve2', 'figure'),
              [Input('land_sea2', 'value'),
               Input('years2', 'value'),
               Input('land_cover_types2', 'value')])
def update_graph2(land_sea, years, land_cover_types):
    fig = make_subplots(rows=1, cols=1, horizontal_spacing=0.1,subplot_titles=['SAF cost curve'])
    for year in years:
        production_potential = calculate_production_potential(results[year],land_cover_types)
        if 'onshore' not in land_sea:
            production_potential = production_potential.loc[production_potential.sea_node]
        if 'offshore' not in land_sea:
            production_potential = production_potential.loc[~production_potential.sea_node]
        cost_curve = calculate_cost_curve(production_potential)

        # fig.append_trace({'x':cost_curve.production_liters_cumsum/1e9,'y':cost_curve.LCOF_liter,'name':'Cost','mode':'lines','type':'scatter','line':{'color':"#4285F4"}},1,1)
        fig.append_trace({'x':cost_curve.production_liters_cumsum/1e9,'y':cost_curve.LCOF_liter,'name':year,'mode':'lines','type':'scatter'},1,1)
    return fig

@app.callback(Output('cost_curve3', 'figure'),
              [Input('land_sea3', 'value'),
               Input('years3', 'value'),
               Input('land_cover_types3', 'value')])
def update_graph3(land_sea, years, land_cover_types):
    fig = make_subplots(rows=1, cols=1, horizontal_spacing=0.1,subplot_titles=['SAF cost curve'])
    for year in years:
        production_potential = calculate_production_potential(results[year],land_cover_types)
        if 'onshore' not in land_sea:
            production_potential = production_potential.loc[production_potential.sea_node]
        if 'offshore' not in land_sea:
            production_potential = production_potential.loc[~production_potential.sea_node]
        cost_curve = calculate_cost_curve(production_potential)

        # fig.append_trace({'x':cost_curve.production_liters_cumsum/1e9,'y':cost_curve.LCOF_liter,'name':'Cost','mode':'lines','type':'scatter','line':{'color':"#4285F4"}},1,1)
        fig.append_trace({'x':cost_curve.production_liters_cumsum/1e9,'y':cost_curve.LCOF_liter,'name':year,'mode':'lines','type':'scatter'},1,1)
    return fig

app.run_server(debug=True)  # Turn off reloader if inside Jupyter