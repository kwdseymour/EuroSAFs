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
    return fig, axes, cb_ax

def correct_CAPEX(df):
    '''
    Prior to commit dac648b, in line 424 of plant_optimizer.py, the capacity units were converted to tons per hour and then multiplied by a CAPEX value that was in tons per annum. 
    This corrects the CAPEX_EUR field with the appropriate unit conversion in the CO2 CAPEX.
    '''
    erroneous_CO2_CAPEX = df['CO2_capture_tonph'] * df['CO2_CAPEX']   
    correct_CO2_CAPEX = df['CO2_capture_tonph'] * 8760 * df['CO2_CAPEX']
    df['CAPEX_EUR'] = df['CAPEX_EUR'] - erroneous_CO2_CAPEX + correct_CO2_CAPEX
    return df

def load_from_path(on_results_path,off_results_path=None,countries=None,file_idx=None):

    # Prior to commit dac648b, in line 424 of plant_optimizer.py, the capacity units were converted to tons per hour and then multiplied by a CAPEX value that was in tons per annum. 
    # Results generated prior to that commit need to correct the CAPEX_EUR field with the appropriate unit conversion in the CO2 CAPEX.
    # The following list was compiled to identify such commits
    bad_commits = [
        '4298cba','d0e9f47','329c557','5620cc1','4433705','cda4835','ec6297d','67ff97c','7ef923a','63c0a64','b6d3813','0906cf7','324f0a7','f38cb0e','a10beae','29efd23','c818762',
        '39ccf4b','088fce1','b82f3c5','8b6e250','bae385c','7288d9f','9a994df','171434d','cbdb7d1','7ce8db9','ad86864','e3f4471','89bbc98','5336c09','a97f320','75a55f5','9764066',
        '6d461f2','1c98517','89ca84f','32af5fd','8e6aca7','a9f79c5','adee9b7','cc1a3e1','ce37cea','8c646f4','eff3ebf','aa3c448','c22b6eb','9de483e','58f79cc','aac7d2e','8b7e92c',
        'cc0b53a','8d900c7','06a79b4','4fa3c6e','5978e29','253481e','e95d97e','0d7597c','dc5565c','065f848','be82d7a','76278e1','5b39531','a923b2a','8123581','df3aec0','f9d76a3',
        'fbe965d','c981fc9','a880cb2','715c363','0fffa21','ef49b22','d12812f','366ebbb','05a9428','c40c17e','660260d','22a2805','ac9afa5','cfa688a','e3cb703','2a7e036','2dd9952',
        '8a990a0','ff2181c','c0e299c','a310ca7','20875e0','d7db98d','31e206e','258ca01','8249390','addcd7c','e1f1acf','9b32964','e326cad','cf98798','3d6f1cf','a768b8a','2cf66a4',
        '09522f3','dda1ff7','925a45e','4db60b5','11ab4dd','39d7fdc','e2e0e21','4cd19b2','daa7ecc','74fe99e','2cb66a4','0850862','ea87652','e8b8a2c','4a31a13','4054dc2','3d6b35b',
        '44c9737','b61e543','a091aac','98d840e','f9ce42b','b8cdfca','a6e67b7','6949c6b','2b2d59c','4bc7d9f','adf52dc','0b1e08a','fa8a2fb','008d4f0','9d363eb','36ce150','491a44b',
        '4f4817b','510ccdd','c1c98f1','be673c4','bb46eea','036501f','00d7d33','5f27350','490e437','28af9bf','16d9e8e','e47de8b','fa7ff93','bcd035d','317cb2a','1a80226','0048761',
        'f462b58','01d32f7','1401523','ad498a3','64f3dfd','8812015','b95779a','f98bef5','5e4ba58','2c7deb6','1b2a4ce','3656abb','f9dd2f3','9340b25','9dfb332','89eaa68','6c3cfa7',
        '0b56d25','c09dbb9','f15d822','8b153a0','79132fc','3abdc3b','bf0c9d3','8e38373','6f0682d','72cec77','d8300f0','41198d6','c3528d9','833f001','07bb209','bfb5e66','a3c11c1',
        '0070f33']

    if off_results_path is None:
        off_results_path = os.path.join(on_results_path,'offshore')
    elif 'offshore' not in off_results_path:
        off_results_path = os.path.join(off_results_path,'offshore')
    
    if 'onshore' not in on_results_path:
        on_results_path = os.path.join(on_results_path,'onshore')

    correct_onshore = any([x in on_results_path for x in bad_commits])
    correct_offshore = any([x in off_results_path for x in bad_commits])
    if correct_onshore:
        print(f'CO2 CAPEX correction applied to onshore files at {on_results_path}.')
    if correct_offshore:
        print(f'CO2 CAPEX correction applied to offshore files at {off_results_path}.')

        
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
        if correct_onshore and len(df)>0:
            df = correct_CAPEX(df)
        
        df_sea = pd.DataFrame(columns=df.columns)
        for i in file_idx:
            for file_glob in glob(f'{off_results_path}/{country}{i}*.csv'): 
                df_sea = df_sea.append(pd.read_csv(file_glob,index_col=0))#.set_index(['lat','lon'])
        df_sea['sea_node'] = True
        if correct_offshore and len(df)>0:
            df = correct_CAPEX(df)
        
        df = df.append(df_sea)
        df['country'] = country
        results = results.append(df)

    europe_grid, europe_borders = load_base_maps()

    results_gdf = gpd.GeoDataFrame(results.merge(europe_grid,on=['lat','lon','country','sea_node'],how='left'))
    return results_gdf

def lcof_map(data,figsize=(15,15),fontsize=35,cmap=None,max_lcof=4,min_lcof=None,missing_kwds=None,legend_kwds={'extend':'max','orientation':'vertical'}):
    
    metrics = {'LCOF_liter':'Levelized cost of fuel [EUR/litre]'}
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