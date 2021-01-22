import geopandas as gpd
import pandas as pd
from statistics import mean
from gurobipy import *

def define_Europe_Coast_Evaluation_Points():
    coast_points = gpd.read_file('./data/Countries_WGS84/processed/Coast_Evaluation_Points.shp')

    eu_points = gpd.read_file('./data/Countries_WGS84/processed/Europe_Evaluation_Points.shp').drop(['PV_lat', 'PV_lon'],axis=1)
    eu_coast_points = gpd.read_file('./data/Countries_WGS84/processed/Europe_Coast_Evaluation_Grid.shp').drop(['geometry', 'PV_lat', 'PV_lon'],axis=1)
    eu_coast_points = eu_coast_points.merge(eu_points)

    new_columns = {'name': [], 'lat': [], 'lon': [], 'next': [], 'weights': []}

    mis_points = []

    for country in list(coast_points['name'].unique()):
        points_coast = coast_points.loc[coast_points['name']==country].copy()
        points_land = eu_coast_points.loc[eu_coast_points['name']==country].copy()

        if len(points_land) > len(points_coast):
            mis_points.append(country)

        for idx_land, point_land in points_land.iterrows():
            new_columns['name'].append(point_land['name'])
            new_columns['lat'].append(point_land['lat'])
            new_columns['lon'].append(point_land['lon'])
            next = []
            weight = []

            for idx_coast, point_coast in points_coast.iterrows():
                next.append([point_coast['lat'], point_coast['lon']])
                weight.append(point_land['geometry'].distance(point_coast['geometry']))

            new_columns['next'].append(next)
            new_columns['weights'].append(weight)

    new_frame = pd.DataFrame(new_columns)
    eu_coast_points = eu_coast_points.merge(new_frame)

    for country in mis_points:
        rounds = len(eu_coast_points.loc[eu_coast_points['name']==country]) - len(coast_points.loc[coast_points['name']==country])
        for i in range(0, rounds):
            point_country = eu_coast_points.loc[eu_coast_points['name'] == country].copy()
            idx = 0
            dis = 0
            for idx_point, point in point_country.iterrows():
                if mean(point['weights']) > dis:
                    idx = idx_point
                    dis = mean(point['weights'])

            eu_coast_points = eu_coast_points.drop(idx)

    eu_coast_points.drop(['next', 'weights'], axis=1).to_file('./data/Countries_WGS84/processed/Europe_Coast_Evaluation_Points.shp')
    eu_coast_points.drop(['geometry'], axis=1).to_pickle('./data/Countries_WGS84/processed/Europe_Coast_Evaluation_Points.pkl')
    eu_coast_points.drop(['geometry'], axis=1).to_csv('./data/Countries_WGS84/processed/Europe_Coast_Evaluation_Points.csv')

def assign_points():
    eu_coast_points = pd.read_pickle('./data/Countries_WGS84/processed/Europe_Coast_Evaluation_Points.pkl')

    new_column = {'name': [], 'lat': [], 'lon': [], 'node': [], 'weight': []}

    for country in list(eu_coast_points['name'].unique()):
        country_points = eu_coast_points.loc[eu_coast_points['name']==country].copy()

        m = Model(name="Points Assignment")

        x = {}

        for idx_01, point in country_points.iterrows():
            for idx_02, node in enumerate(point['next']):
                name = str(point['lat'])+'_'+str(point['lon'])+'_'+str(node[0])+'_'+str(node[1])
                x[name] = m.addVar(vtype=GRB.BINARY, obj=point['weights'][idx_02], name="x_"+str(point['lat'])+'_'+str(point['lon'])+'_'+str(node[0])+'_'+str(node[1]))

        m.update()

        for idx_01 in country_points['next'].iloc[0]:
            m.addConstr( quicksum(x[str(i)+'_'+str(country_points['lon'].to_list()[idx])+'_'+str(idx_01[0])+'_'+str(idx_01[1])] for idx, i in enumerate(country_points['lat'].to_list())) <= 1 )

        for idx_01, point in country_points.iterrows():
            m.addConstr( quicksum( x[str(point['lat'])+'_'+str(point['lon'])+'_'+str(i[0])+'_'+str(i[1])] for i in point['next'] ) == 1 )

        m.update()

        obj = LinExpr()

        for idx_01, point in country_points.iterrows():
            for idx_02, node in enumerate(point['next']):
                name = str(point['lat'])+'_'+str(point['lon'])+'_'+str(node[0])+'_'+str(node[1])
                obj.addTerms(point['weights'][idx_02], x[name])

        m.setObjective(obj, sense=GRB.MINIMIZE)

        m.update()

        # Call the optimizer
        m.optimize()

        for entry in x.values():
            if entry.x == 1:
                name = entry.varName.strip('x_').split('_')
                new_column['name'].append(country)
                new_column['lat'].append(float(name[0]))
                new_column['lon'].append(float(name[1]))
                new_column['node'].append([float(name[2]), float(name[3])])
                new_column['weight'].append(entry.Obj)

        m.dispose()

    new_frame = pd.DataFrame(new_column)
    eu_coast_points = eu_coast_points.merge(new_frame)

    eu_coast_points.to_pickle('./data/Countries_WGS84/processed/Europe_Coast_Evaluation_Points.pkl')
    eu_coast_points.to_csv('./data/Countries_WGS84/processed/Europe_Coast_Evaluation_Points.csv')

define_Europe_Coast_Evaluation_Points()
assign_points()