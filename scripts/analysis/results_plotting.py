import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_base_maps():
    europe_grid = gpd.read_file('../../data/Countries_WGS84/processed/Europe_Evaluation_Grid.shp')
    europe_grid.rename(columns={'grid_lat':'lat','grid_lon':'lon'},inplace=True)
    europe_grid['sea_node'] = europe_grid.sea_node.astype(bool)
    europe_borders = gpd.read_file('../../data/Countries_WGS84/processed/Europe_WGS84.shp')
    return europe_grid, europe_borders

EU_EFTA = load_base_maps()[0].country.unique()

def plot_results(data,countries,metrics,figsize=(25,20),ncols=None,fontsize=None,cmap=None,vmax_dict={},vmin_dict={},missing_kwds=None,legend_kwds={}, show_axis=False):
    if ncols == None:
        ncols = int(np.ceil(np.sqrt(len(metrics))))
    nrows = int(np.ceil(len(metrics)/ncols))
    fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)
    axes = np.reshape(axes,(1,nrows*ncols))[0]

    if fontsize == None:
        fontsize = figsize[0]*2
    df = data.loc[data.country.isin(countries)].copy()

    europe_grid, europe_borders = load_base_maps()

    borders = europe_borders.loc[europe_borders.country.isin(countries)].copy()
    if cmap is None:
        cmap = 'RdYlGn_r'
    for i,(metric,description) in enumerate(metrics.items()):
        vmax = vmax_dict[metric] if metric in vmax_dict.keys() else None
        vmin = vmin_dict[metric] if metric in vmin_dict.keys() else None
        if metric not in ['turbine_type','specific_capacity_class']:
            l_kwds = legend_kwds.copy()
            l_kwds.setdefault('orientation','horizontal')
            l_kwds['ax'] = axes[i]
            l_kwds.pop('bbox_to_anchor',None)
            l_kwds.pop('fontsize',None)
            if l_kwds['orientation'] == 'vertical':
                divider = make_axes_locatable(axes[i])
                cax = divider.append_axes("right", size="3%", pad=-2)
            else:
                cax = None
        else:
            l_kwds = legend_kwds.copy()
            l_kwds.pop('orientation',None)
            l_kwds.setdefault('loc','upper center')
            l_kwds.setdefault('ncol',min(3,df[metric].nunique()))
            l_kwds.setdefault('fontsize',fontsize*.75)
            cax = None
            if df[metric].nunique()>10:
                cmap = 'tab20'
        if not any(df[metric].isna()):
            m_kwds = None
        else:
            m_kwds = missing_kwds
        df.plot(column=metric,legend=True,cmap=cmap,vmin=vmin,vmax=vmax,missing_kwds=m_kwds,legend_kwds=l_kwds,cax=cax,ax=axes[i])
        borders.boundary.plot(color='k',ax=axes[i])
        cb_ax = axes[i].figure.axes[-1]
        cb_ax.tick_params(labelsize=fontsize)
        axes[i].tick_params(labelsize=fontsize)
        if not show_axis:
            axes[i].axis('off')
        axes[i].set_title(description,fontsize=fontsize)
#     plt.tight_layout()

def load_from_path(on_results_path,off_results_path=None,countries=None,file_idx=None):
    if off_results_path is None:
        off_results_path = os.path.join(on_results_path,'offshore')
    elif 'offshore' not in off_results_path:
        off_results_path = os.path.join(off_results_path,'offshore')
    
    if 'onshore' not in on_results_path:
        on_results_path = os.path.join(on_results_path,'onshore')
        
    if countries is None:
        countries = EU_EFTA
    elif isinstance(countries,str):
        countries =[countries]
        
    if file_idx is None:
        file_idx = ['']
    elif isinstance(file_idx,str) or isinstance(file_idx,int):
        file_idx = [str(file_idx)]
    file_idx = ['_'+x for x in file_idx]
        
    results = pd.DataFrame()
        
    for country in countries:
        df = pd.DataFrame()
        for i in file_idx:
            for file_glob in glob(f'{on_results_path}/{country}{i}*.csv'): 
                df = df.append(pd.read_csv(file_glob,index_col=0))#.set_index(['lat','lon'])
        df['sea_node'] = False
        
        df_sea = pd.DataFrame(columns=df.columns)
        for i in file_idx:
            for file_glob in glob(f'{off_results_path}/{country}{i}*.csv'): 
                df_sea = df_sea.append(pd.read_csv(file_glob,index_col=0))#.set_index(['lat','lon'])
        df_sea['sea_node'] = True
        
        df = df.append(df_sea)
        df['country'] = country
        results = results.append(df)
        
    europe_grid, europe_borders = load_base_maps()

    results_gdf = gpd.GeoDataFrame(results.merge(europe_grid,on=['lat','lon','country','sea_node'],how='left'))
    return results_gdf

def lcof_map(data,figsize=(15,15),fontsize=35,cmap=None,max_lcof=4,min_lcof=None,missing_kwds=None,legend_kwds={'extend':'max','orientation':'vertical'}):
    
    metrics = {'LCOF_liter':'Levelized cost of fuel [EUR/liter]'}
    vmax_dict = {'LCOF_liter':max_lcof}
    vmin_dict = {'LCOF_liter':min_lcof}

    plot_results(data=data,countries=data.country.unique(),metrics=metrics,cmap=cmap,vmax_dict=vmax_dict,vmin_dict=vmin_dict,legend_kwds=legend_kwds,figsize=figsize,fontsize=fontsize)
    plt.tight_layout()

def plot_country(country,metrics,onshore=True,offshore=False,**plot_results_kwds):
    results = pd.DataFrame()
    if onshore:
        for file_glob in glob(f'{results_path}/onshore/{country}*.csv'): 
            results = results.append(pd.read_csv(file_glob,index_col=0))
    if offshore:
        for file_glob in glob(f'{results_path}/offshore/{country}*.csv'): 
            results = results.append(pd.read_csv(file_glob,index_col=0))
    results['country'] = country

    europe_grid, europe_borders = load_base_maps()

    results_gdf = gpd.GeoDataFrame(results.merge(europe_grid,on=['lat','lon','country'],how='left'))
    results_gdf = results_gdf.loc[~results_gdf.sea_node]
    plot_results(data=results_gdf,metrics=metrics,countries=[country],**plot_results_kwds)