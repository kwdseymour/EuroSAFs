# In this script, the available land area for PV/wind installations of each evaluation point/cell in each country is determined using the Copernicus Land Monitoring Service CORINE Land Cover database.
# Land cover data: https://land.copernicus.eu/pan-european/corine-land-cover/clc2018
# Elevation data: https://land.copernicus.eu/imagery-in-situ/eu-dem/eu-dem-v1.1
# Land cover documentation: https://land.copernicus.eu/user-corner/technical-library/clc-product-user-manual
# In a first step, the land area (in squre kilometers) covered by each pixel in the LC raster is calculated. It is a function of latitude.
# Then, based on the land types/attitude chosen as valid (it is assumed that only certain land types are available for PV/wind installations), each pixel is either assigned 1 or 0,
# indicating its availability. For available pixels, the pixel's land area is then assigned. For unavailable pixels, a value of 0 is assigned.
# In the last step, the land availability rasters created before are resampled to the lower MERRA resolution (the available land area within each MERRA evaluation cell is determined from these rasters).
#
# Required Data:
# - CORINE Land Cover Raster (Overview over land types in ./data/CORINE_legend.xls)
#   ./data/CORINE_land_cover_data/EU_LandType_trans.tif
# - Altitude Data for Europe
#   ./data/altitude_data/*.TIF
# - Shape file of Europe
#   ./data/Countries_WGS84/processed/Europe_WGS84.shp
# - Shape file of grided europe
#   './data/Countries_WGS84/processed/Europe_Evaluation_Grid.shp'
#
# Produced Files:
# - Raster File for each country
#   ./data/land_cover_rasters/01_land_scenario/*.tif'
# - Shape/CSV file with the results for each country
#   ./data/Countries_WGS84/Land_Availability/01_land_scenario/Land_Availability_' + country + '_AvailLand_Points.shp
#   ./results/00_land_availability/' + country + '_land_availability.csv

# Setup
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from scipy.ndimage.measurements import label
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
import rasterio.merge
from scipy.interpolate import RegularGridInterpolator
import glob

# Methods
def mask_lc_by_country(lc_data,europe_shapes,country):
    '''Masks the given land cover raster (lc_data) using the country geometry found in the europe_shapes GeoDataFrame.'''
    masked_lc, masked_lc_transform = rasterio.mask.mask(lc_data, [europe_shapes.loc[europe_shapes.name==country,'geometry'].item()], crop=True)
    return masked_lc, masked_lc_transform

def filter_land_types(lc_array,valid_land_types):
    '''Assigns a value of 1 or 0 to each pixel depending on whether the pixel is a valid land type (defined in valid_land_types list)
    Returns an array with the same shape as the input array (lc_array)'''
    land_mask = np.logical_or.reduce(tuple([lc_array==x for x in valid_land_types]))
    land_availability = np.where(land_mask,1,0)
    labeled_array, num_features = label(land_availability)
    return land_availability[0], labeled_array, num_features

def filter_attitude(at_array,valid_attitude):
    '''Assigns a value of 1 or 0 to each pixel depending on whether the pixel is a valid land type (defined in valid_land_types list)
    Returns an array with the same shape as the input array (lc_array)'''
    land_mask = np.logical_or.reduce(tuple([at_array<=valid_attitude]))
    land_availability = np.where(land_mask,1,0)
    labeled_array, num_features = label(land_availability)
    return land_availability[0], labeled_array, num_features

def get_area_array(raster,transform):
    '''Creats an array wtih the same shape as the input array containing the land area of each pixel.'''
    res = -transform[4] # resolution in degrees
    height = raster[0].shape[0]
    width = raster[0].shape[1]
    upper_lat = transform[5] - res/2
    lower_lat = transform[5] - res*height
    lat_vector = np.arange(upper_lat,lower_lat,-res)

    def pixel_area(latitude):
        '''Calculates the area of a pixel at a given latitude (degrees) with a given resolution (degrees) in square kilometers.
        https://en.wikipedia.org/wiki/Geographic_coordinate_system'''
        lat_distance = (111132.954 - 559.82*np.cos(2*np.radians(latitude)) + 1.175*np.cos(4*np.radians(latitude)) - 0.0023*np.cos(6*np.radians(latitude)))/1000*res
        lon_distance = (111412.84*np.cos(np.radians(latitude)) - 93.5*np.cos(3*np.radians(latitude)) + 0.118*np.cos(5*np.radians(latitude)))/1000*res
        area = lat_distance * lon_distance
        return area # [km2]
    pa = np.vectorize(pixel_area)
    area_vector = pa(lat_vector)
    area_array = np.array([area_vector,]*width).transpose()
    return area_array #[values in km2]

def generate_lc_raster(lc_data,europe_shapes,country,valid_land_types, frames, valid_attitude):
    '''Creates a land availability raster indicating the available land area within each pixel for the given country. NaN's indicate no available land area.
    - lc_data should be the land cover raster downloaded from the USGS
    - europe_shapes should be a GeoDataFrame containing the borders for European countries
    - country respresents the country that will be evaluated
    - valid_land_types is the set land types according to the USGS data notation that are considered valid for PV/wind installations
    '''
    def regrid(data, out_x, out_y):
        m = max(data.shape[0], data.shape[1])
        y = np.linspace(0, 1.0 / m, data.shape[0])
        x = np.linspace(0, 1.0 / m, data.shape[1])
        interpolating_function = RegularGridInterpolator((y, x), data)
        yv, xv = np.meshgrid(np.linspace(0, 1.0 / m, out_y), np.linspace(0, 1.0 / m, out_x))
        return interpolating_function((xv, yv))

    masked_lc, masked_lc_transform = mask_lc_by_country(lc_data, europe_shapes, country)
    i = 0
    for path in frames:
        data = rasterio.open(path, 'r')
        try:
            masked_at, masked_at_transform = mask_lc_by_country(data, europe_shapes, country)
            if np.max(masked_at) < -1000:
                i += 1
                if i == len(frames):
                    print('No matching dataset found.')
                    exit()
                continue
            else:
                print('Dataset found for ' + country + '. Max. Altitude: ' + str(np.max(masked_at)) + '. Min. Altitude: ' + str(np.min(masked_at)))
                break
        except ValueError:
            i += 1
            if i == len(frames):
                print('No matching dataset found.')
                exit()
            continue

    land_avail_lc, labeled_array_lc, num_features_lc = filter_land_types(masked_lc, valid_land_types)
    land_avail_at, labeled_array_at, num_features_at = filter_attitude(masked_at, valid_attitude)
    land_avail_at = regrid(land_avail_at, land_avail_lc.shape[0], land_avail_lc.shape[1])
    land_avail_at[land_avail_at >= 1] = 1
    land_avail_at[land_avail_at < 1] = 0
    land_avail_lc[land_avail_at == 0] = 0
    area_array = get_area_array(masked_lc,masked_lc_transform)
    area_avail_array = np.multiply(land_avail_lc,area_array)
    print('Areas defined.')
    return area_avail_array, masked_lc_transform, labeled_array_lc, num_features_lc

def save_lc_raster(array,transform,country_name):
    '''Saves the raster created using the generate_lc_raster function to a file.'''
    output_metadata = {'driver': 'GTiff',
                   'dtype':'float32',
                   'nodata':None,
                   'width':array.shape[1],
                   'height':array.shape[0],
                   'count':1,
                   'crs': rasterio.crs.CRS.from_epsg(4326),
                   'transform':transform}
    with rasterio.open('./data/land_cover_rasters/01_land_scenario/'+country_name+'.tif','w',**output_metadata) as file:
        file.write(np.array([array]).astype(rasterio.float32))

def available_area_grid(country, labeled_array, num_features, transform, grouped):
    '''For the country provided, this function opens the corresponding land availability raster and determines the available land area (in squre kilometers)
    within each cell (evaluation point) of the country's gridded geometry.

    Returns a GeoDataFrame with the result.'''
    if grouped == True:
        country_grid = europe_grid.loc[europe_grid.name == country].copy().reset_index(drop=True)

        area_list = pd.DataFrame()

        for row in country_grid.iterrows():
            create_area_list(row, labeled_array=labeled_array, num_features=num_features, transform=transform)
            area = pd.read_pickle('./data/land_cover_rasters/01_land_scenario/Temp/temp_' + '_' + str(row[1][1]) + '_' + str(row[1][2]) + '_' + country + '.pkl')
            area_list = area_list.append(area)

        if len(area_list) != 0:
            country_grid = country_grid.merge(area_list)
            country_grid = country_grid[['lat', 'lon', 'avail_area_sqkm', 'geometry']]
        else:
            country_grid = pd.DataFrame()

    else:
        dataLC = rasterio.open('./data/land_cover_rasters/01_land_scenario/' + country + '.tif')
        country_grid = europe_grid.loc[europe_grid.name == country].copy().reset_index(drop=True)
        country_grid['avail_area_sqkm'] = country_grid.apply(lambda x: np.nansum(rasterio.mask.mask(dataLC, [x['geometry']], crop=True)[0]), axis=1)
        country_grid = country_grid[['lat', 'lon', 'avail_area_sqkm', 'geometry']]

    return country_grid

def create_area_list(row, labeled_array, num_features, transform):
    '''For each evaluation point in each country the the biggest group of connected areas is defined and temporarly saved.'''
    print('Start with point ' + str(row[1][1]) + ' , ' + str(row[1][2]) + '.')
    space = 0

    for i in range(1, num_features+1):
        dataLC_set_up = rasterio.open('./data/land_cover_rasters/01_land_scenario/' + country + '.tif')
        data = dataLC_set_up.read()
        data[labeled_array != i] = 0
        data = data.squeeze()
        output_metadata = {'driver': 'GTiff',
                           'dtype': 'float32',
                           'nodata': None,
                           'width': data.shape[1],
                           'height': data.shape[0],
                           'count': 1,
                           'crs': rasterio.crs.CRS.from_epsg(4326),
                           'transform': transform}
        with rasterio.open('./data/land_cover_rasters/01_land_scenario/Temp/temp_' + country + '_' + str(row[1][1]) + '_' + str(row[1][2]) + '.tif', 'w', **output_metadata) as file:
            file.write(np.array([data, ]).astype(rasterio.float32))

        dataLC = rasterio.open('./data/land_cover_rasters/01_land_scenario/Temp/temp_' + country + '_' + str(row[1][1]) + '_' + str(row[1][2]) + '.tif')
        raster_area = rasterio.mask.mask(dataLC, [row[1][5]], crop=True)[0]
        if raster_area.sum() > space:
            space = raster_area

    d = {'lat':[row[1][1]], 'lon':[row[1][2]], 'avail_area_sqkm':[space]}
    area = pd.DataFrame(data=d)
    area.to_pickle('./data/land_cover_rasters/01_land_scenario/Temp/temp_' + '_' + str(row[1][1]) + '_' + str(row[1][2]) + '_' + country + '.pkl')

def change_reference_system(file):
    '''Changes the coordinate reference system of the imported altitude data to match the already existing shape files'''
    dst_crs = 'EPSG:4326'
    with rasterio.open(file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(os.path.splitext(file)[0] + '_trans.TIF' , 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def merge_frames():
    '''To cover each country completly with one file the different altitude data is combined to a larger file and saved again'''
    for x in range(30, 50, 10):
        for y in range(10, 50, 10):
            print('Merge E' + str(x + 5) + 'N' + str(y + 5))
            dst_crs = 'EPSG:4326'
            file1 = rasterio.open('./data/Altitude_Data/eu_dem_v11_E' + str(x) + 'N' +str(y) + '_trans.TIF', 'r')
            file2 = rasterio.open('./data/Altitude_Data/eu_dem_v11_E' + str(x) + 'N' + str(y + 10) + '_trans.TIF', 'r')
            file3 = rasterio.open('./data/Altitude_Data/eu_dem_v11_E' + str(x + 10) + 'N' + str(y) + '_trans.TIF', 'r')
            file4 = rasterio.open('./data/Altitude_Data/eu_dem_v11_E' + str(x + 10) + 'N' + str(y + 10) + '_trans.TIF', 'r')

            combination, out_trans = rasterio.merge.merge([file1, file2, file3, file4])
            out_meta = file1.meta.copy()
            out_meta.update({"driver": "GTiff",
                             "height": combination.shape[1],
                             "width": combination.shape[2],
                             "transform": out_trans,
                             "crs": dst_crs})
            with rasterio.open(('./data/Altitude_Data/eu_dem_v11_E' + str(x + 5) + 'N' + str(y + 5) + '.TIF'), "w", **out_meta) as dest:
                dest.write(combination)
                print('Done.')

def pre_process_altitude_data():
    '''Prepares the altitude data for further evaluation by changing the reference system and change the size of each file.'''
    for file in glob.glob('./data/Altitude_Data/*.TIF'):
        print('Start ' + file)
        change_reference_system(file)
    merge_frames()

def through_every_country(country):
    '''Runs the available area evaluation for each country and save the results in a shape/csv file.'''
    print('Start ' + country)
    country_area_avail = gpd.GeoDataFrame({'name': [], 'lat': [], 'lon': [], 'geometry': [], 'avail_area_sqkm': []})
    area_avail_array,transform, labeled_array, num_features = generate_lc_raster(lc_data,europe,country,valid_land_types, frames, valid_attitude)
    save_lc_raster(area_avail_array,transform,country)
    available_area = available_area_grid(country, labeled_array, num_features, transform, True)
    country_area_avail = pd.concat([country_area_avail, available_area])
    if country_area_avail.empty != True:
        country_area_avail['name'] = country
        country_area_avail.to_file('./data/Countries_WGS84/Land_Availability/01_land_scenario/Land_Availability_' + country + '_AvailLand_Points.shp')
        country_area_avail.drop(['geometry'], 1).to_csv('./results/00_land_availability/' + country + '_land_availability.csv')
        print(country + ' saved.')
    else:
        print('No areas found.')

# Pre-processing of the altitude: Only execute once.
#pre_process_altitude_data()

# Definition of the valid land types
# These correspond to the types described in Annex 1 of the land cover documentation linked in the top of this script
valid_land_types = [29, 32]

# Considered Tree Line
valid_attitude = 1800

# Shape file of europe
europe = gpd.read_file('./data/Countries_WGS84/Europe_WGS84.shp')

# Shape file of grided europe
europe_grid = gpd.read_file('./data/Countries_WGS84/Europe_Evaluation_Grid.shp')

# TIF file of land use raster
lc_data = rasterio.open('./data/CORINE_land_cover_data/EU_LandType_trans.tif')

# Change of refernce system for the land type data. Only execute once
#dst_crs = 'EPSG:4326'
#transform, width, height = calculate_default_transform(lc_data.crs, dst_crs, lc_data.width, lc_data.height, *lc_data.bounds)
#kwargs = lc_data.meta.copy()
#kwargs.update({'crs': dst_crs,'transform': transform,'width': width,'height': height})
#with rasterio.open('Elevation_trans.tif' , 'w', **kwargs) as dst:
    #for i in range(1, lc_data.count + 1):reproject(
                    #source=rasterio.band(lc_data, i),
                    #destination=rasterio.band(dst, i),
                    #src_transform=lc_data.transform,
                    #src_crs=lc_data.crs,
                    #dst_transform=transform,
                    #dst_crs=dst_crs,
                    #resampling=Resampling.nearest)

# List of TIF files of attitude raster
frames = []
frames.append('./data/Altitude_Data/eu_dem_v11_E60N10_trans.TIF')
for x in range(20, 50, 10):
    for y in range(10, 50, 10):
        frames.append('./data/Altitude_Data/eu_dem_v11_E' + str(x + 5) + 'N' + str(y + 5) +'.TIF')

# List of countries to evaluate
countries_filepath = './data/EU_EFTA_Countries.csv'
EU_EFTA = list(pd.read_csv(countries_filepath, index_col=0)['country'])

# Evaluate each country
for country in EU_EFTA:
        through_every_country(country)