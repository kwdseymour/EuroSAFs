import pandas as pd
import numpy as np
import geopandas as gpd

europe_grid = gpd.read_file('../../data/Countries_WGS84/processed/Europe_Evaluation_Grid.shp')
europe_grid.rename(columns={'grid_lat':'lat','grid_lon':'lon'},inplace=True)
europe_grid['sea_node'] = europe_grid.sea_node.astype(bool)

sea_grid = europe_grid.loc[europe_grid.sea_node].reset_index()
sea_grid['avail_area_sqkm'] = np.nan

for country in sea_grid.country.unique():
    country_sea_grid = sea_grid.loc[sea_grid.country==country].copy()
    if len(country_sea_grid) == 0:
        print(country,'complete.')
        continue
    for idx in country_sea_grid.index:
        row = country_sea_grid.loc[[idx]]
        row = row.to_crs(f'+proj=cea +lat_0={row.iloc[0]["lat"]} +lon_0={row.iloc[0]["lon"]} +units=m')
        country_sea_grid.loc[idx,'avail_area_sqkm'] = row.iloc[0]['geometry'].area/1e6
    
    country_sea_grid[['country','lat','lon','avail_area_sqkm']].to_csv(f'../../results/land_availability/sea_area/{country}.csv',index=False)
    print(country,'complete.')