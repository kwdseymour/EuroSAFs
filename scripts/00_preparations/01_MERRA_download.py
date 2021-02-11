'''
This notebook is used to download files from [MERRA-2](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/) (referred to as MERRA).

The MERRA dataset used, [M2T1NXSLV.5.12.4](https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_5.12.4/summary) ([DOCUMENTATION](https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf)), contains historic northward and eastward wind components at multiple heights for locations across the globe. The resolution of the MERRA grid is 0.5 x 0.625°. In this notebook, the wind speeds at 10 m (U/V10M corresponds to 10 meters above the zero-plane displacement height) and 50 m (U/V50M corresponds to 50 meters above Earth's surface) are extracted, as well as the zero-plane displacement height (DISPH). 

The code is adapted from the [weather_data](https://github.com/Open-Power-System-Data/weather_data) package and MERRA's [help page](https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Use%20the%20Web%20Services%20API%20for%20Subsetting).

For each country given, the nearest MERRA point is identified and the hourly data for all points in each day of 2016 is downloaded to a file in a folder named according to the country.
'''

# python scripts/00_preparations/01_MERRA_download.py
import numpy as np
import logging
import multiprocessing as mp
import geopandas as gpd
import urllib
import getpass
import requests
import os
from http.cookiejar import CookieJar
import time
from calendar import monthrange
import json
import sys

# Get EuroSAFs parent directory 
SAF_directory = os.path.dirname(__file__)
for i in range(2):
    SAF_directory = os.path.dirname(SAF_directory)
# Get scratch path
if 'cluster/home' in os.getcwd():
    # Uses the $SCRATCH environment variable to locate the scratch file if this module is run within Euler
    scratch_path = os.environ['SCRATCH']
else:
    scratch_path = os.path.join(SAF_directory,'scratch')
# Read configuration file. The MERRA username is extracted from here.
with open(os.path.join(SAF_directory,'scripts/config.json')) as config_file:
    config = json.load(config_file)

# Add a logger
sys.path.append(os.path.join(SAF_directory,'scripts/03_plant_optimization'))
from plant_optimization.utilities import create_logger
logger = create_logger(scratch_path,__name__,__file__)

# Download raw data
#This part defines the input parameters according to the user and creates an URL that can download the desired MERRA-2 data via the OPeNDAP interface (see <a href="documentation.ipynb">documentation notebook</a> for information on OPeNDAP).


'''
Definition of desired coordinates. The user has to input two corner coordinates 
of a rectangular area (Format WGS84, decimal system).
- Southwest coordinate: lat_1, lon_1
- Northeast coordinate: lat_2, lon_2

The area/coordinates will be converted from lat/lon to the MERRA-2 grid coordinates.
Since the resolution of the MERRA-2 grid is 0.5 x 0.625°, the given exact coordinates will 
matched as close as possible.

------
User input of coordinates
------
Example: Germany (lat/lon)
Northeastern point: 55.05917°N, 15.04361°E
Southwestern point: 47.27083°N, 5.86694°E

It is important to make the southwestern coordinate lat_1 and lon_1 since
the MERRA-2 portal requires it!
'''

def translate_lat_to_geos5_native(latitude):
    """
    The source for this formula is in the MERRA2 
    Variable Details - File specifications for GEOS pdf file.
    The Grid in the documentation has points from 1 to 361 and 1 to 576.
    The MERRA-2 Portal uses 0 to 360 and 0 to 575.
    latitude: float Needs +/- instead of N/S
    """
    return ((latitude + 90) / 0.5)

def translate_lon_to_geos5_native(longitude):
    """See function above"""
    return ((longitude + 180) / 0.625)

def find_closest_coordinate(calc_coord, coord_array):
    """
    Since the resolution of the grid is 0.5 x 0.625, the 'real world'
    coordinates will not be matched 100% correctly. This function matches 
    the coordinates as close as possible. 
    """
    # np.argmin() finds the smallest value in an array and returns its
    # index. np.abs() returns the absolute value of each item of an array.
    # To summarize, the function finds the difference closest to 0 and returns 
    # its index. 
    index = np.abs(coord_array-calc_coord).argmin()
    return coord_array[index]

def get_MERRA_coordinates(SW_coord,NE_coord):
    '''
    Converts WGS-84 coordinates to the CRS of the MERRA-2 grid.
    The user has to input two corner coordinates of a rectangular area (Format WGS84, decimal system).

    SW_coord: Southwest coordinate: (lat_2, lon_2)
    NE_coord: Northeast coordinate: (lat_1, lon_1)
    
    The area/coordinates will be converted from lat/lon to the MERRA-2 grid coordinates. 
    Since the resolution of the MERRA-2 grid is 0.5 x 0.625°, the given exact coordinates will matched as close as possible.'''
    
    # Southwestern coordinate
    lat_1 = SW_coord[0]; lon_1 = SW_coord[1]
    # Northeastern coordinate
    lat_2 = NE_coord[0]; lon_2 = NE_coord[1]
    
    # The arrays contain the coordinates of the grid used by the API.
    # The values are from 0 to 360 and 0 to 575
    lat_coords = np.arange(0, 361, dtype=int)
    lon_coords = np.arange(0, 576, dtype=int)

    # Translate the coordinates that define your area to grid coordinates.
    lat_coord_1 = translate_lat_to_geos5_native(lat_1)
    lon_coord_1 = translate_lon_to_geos5_native(lon_1)
    lat_coord_2 = translate_lat_to_geos5_native(lat_2)
    lon_coord_2 = translate_lon_to_geos5_native(lon_2)


    # Find the closest coordinate in the grid.
    lat_co_1_closest = find_closest_coordinate(lat_coord_1, lat_coords)
    lon_co_1_closest = find_closest_coordinate(lon_coord_1, lon_coords)
    lat_co_2_closest = find_closest_coordinate(lat_coord_2, lat_coords)
    lon_co_2_closest = find_closest_coordinate(lon_coord_2, lon_coords)
    
    return [lat_co_1_closest, lon_co_1_closest, lat_co_2_closest, lon_co_2_closest]

    ## Subsetting data

    # Combining parameter choices above/translation according to OPenDAP guidelines into URL-appendix

def translate_year_to_file_number(year):
    '''
    The file names consist of a number and a meta data string. 
    The number changes over the years. 1980 until 1991 it is 100, 
    1992 until 2000 it is 200, 2001 until 2010 it is  300 
    and from 2011 until now it is 400.
    '''
    file_number = ''
    
    if year >= 1980 and year < 1992:
        file_number = '100'
    elif year >= 1992 and year < 2001:
        file_number = '200'
    elif year >= 2001 and year < 2011:
        file_number = '300'
    elif year >= 2011:
        file_number = '400'
    else:
        raise Exception('The specified year is out of range.')
    
    return file_number
    


def generate_url_params(parameter, time_para, lat_para, lon_para):
    """Creates a string containing all the parameters in query form"""
    parameter = map(lambda x: x + time_para, parameter)
    parameter = map(lambda x: x + lat_para, parameter)
    parameter = map(lambda x: x + lon_para, parameter)
    
    base = ','.join(parameter)
    extension = ',lat{},time{},lon{}'.format(lat_para,time_para,lon_para)
    return base + extension
    
    

def generate_download_links(download_years, base_url, dataset_name, url_params):
    """
    Generates the links for the download. 
    download_years: The years you want to download as array. 
    dataset_name: The name of the data set. For example tavg1_2d_slv_Nx
    """
    urls = {}
    for y in download_years: 
    # build the file_number
        y_str = str(y)
        file_num = translate_year_to_file_number(y)
        for m in range(1,13):
            # build the month string: for the month 1 - 9 it starts with a leading 0. 
            # zfill solves that problem
            m_str = str(m).zfill(2)
            # monthrange returns the first weekday and the number of days in a 
            # month. Also works for leap years.
            _, nr_of_days = monthrange(y, m)
            for d in range(1,nr_of_days+1):
                d_str = str(d).zfill(2)
                # Create the file name string
                file_name = 'MERRA2_{num}.{name}.{y}{m}{d}.nc4'.format(
                    num=file_num, name=dataset_name, 
                    y=y_str, m=m_str, d=d_str)
                # Create the query
                query = '{base}{y}/{m}/{name}.nc4?{params}'.format(
                    base=base_url, y=y_str, m=m_str, 
                    name=file_name, params=url_params)
                urls[file_name] = query
                
    return urls

def generate_download_links_by_day(day_list, base_url, dataset_name, url_params):
    """
    Generates the links for the download. 
    day_list: The days you want to download as array (yyyymmdd). 
    dataset_name: The name of the data set. For example tavg1_2d_slv_Nx
    """
    urls = {}
    for day in day_list:
        y_str = day[:4]
        m_str = day[4:6]
        d_str = day[6:]
        file_num = translate_year_to_file_number(int(y_str))
        # Create the file name string
        file_name = 'MERRA2_{num}.{name}.{y}{m}{d}.nc4'.format(
            num=file_num, name=dataset_name, 
            y=y_str, m=m_str, d=d_str)
        # Create the query
        query = '{base}{y}/{m}/{name}.nc4?{params}'.format(
            base=base_url, y=y_str, m=m_str, 
            name=file_name, params=url_params)
        urls[file_name] = query
    return urls

def generate_urls(MERRA_coords,year): #daylist):
    '''Generates a dictionary of URLs according to MERRA OPeNDAP protocol to download each the desired subsetted data 
    files for each day in the time span given.
    '''
    # Creates a string that looks like [start:1:end]. start and end are the lat or
    # lon coordinates define your area.
    requested_lat = '[{lat_1}:{lat_2}]'.format(lat_1=MERRA_coords[0], lat_2=MERRA_coords[2])
    requested_lon = '[{lon_1}:{lon_2}]'.format(lon_1=MERRA_coords[1], lon_2=MERRA_coords[3])

    requested_time = '[0:23]'
    
    # Generate wind URLs
    
    # Parameter definitions: https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf
    requested_params = ['U10M','V10M','U50M','V50M', 'DISPH']
    parameter = generate_url_params(requested_params, requested_time,
                                    requested_lat, requested_lon)
    BASE_URL = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/'
#     generated_URLs['wind'] = generate_download_links_by_day(daylist, BASE_URL, 'tavg1_2d_slv_Nx', parameter)
    generated_URLs = generate_download_links(year, BASE_URL, 'tavg1_2d_slv_Nx', parameter)

    return generated_URLs

def establish_connection(username=config['merra_username'],password=config['merra_password']):
    '''An Earthdata account is required to download data. An account can be created here: https://urs.earthdata.nasa.gov/
    
    This function creates a password manager to deal with the 401 response that is returned from Earthdata Login.
    '''

    password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)

    # Create a cookie jar for storing cookies. This is used to store and return the session cookie #given to use by the data server
    cookie_jar = CookieJar()

    # Install all the handlers.
    opener = urllib.request.build_opener (urllib.request.HTTPBasicAuthHandler (password_manager),urllib.request.HTTPCookieProcessor (cookie_jar))
    urllib.request.install_opener(opener)
    
def download_files(generated_URLs,download_path):
    '''Open a request for the data, and download files'''
    
    found_files = 0
    for file_name,URL in generated_URLs.items():
        path = os.path.join(download_path,file_name)
        if os.path.isfile(path):
            found_files += 1
        else:
            DataRequest = urllib.request.Request(URL)
            DataResponse = urllib.request.urlopen(DataRequest)

        # Print out the result
            DataBody = DataResponse.read()

        # Save file to working directory
            try:
                file_ = open(path, 'wb')
                file_.write(DataBody)
                file_.close()
            except requests.exceptions.HTTPError as e:
                 logger.error(e)
    if found_files > 0:
        logger.info('{} files were already found in {} and were therefore not downloaded.'.format(found_files,download_path))

def download_subset(SW_coord,NE_coord,year,download_path):
    '''
    Downloads the required data files for the geographic and temporal subset provided from MERRA-2.
    
    SW_coord: Southwest coordinate: (lat_1, lon_1)
    NE_coord: Northeast coordinate: (lat_2, lon_2)
    day_list: The days you want to download as array (yyyymmdd). 
    '''
    
    MERRA_coords = get_MERRA_coordinates(SW_coord,NE_coord)
    generated_URLs = generate_urls(MERRA_coords,year)
    while True:
        try:
            download_files(generated_URLs,download_path)
            break
        except:
            logger.info('Connection failed. Retrying...')
            establish_connection()
    
# Import the Europe country points geodataframe used to extract the MERRA points found within each country
# This file was created in the Notebook: N01_country_boundaries.ipynb
europe_merra_points = gpd.read_file(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Europe_MERRA_Evaluation_Points.shp'))
# coast_points = gpd.read_file(os.path.join(SAF_directory,'data/Countries_WGS84/processed/Coast_Evaluation_Points.shp'))

def download_country_files(country,year=2016):
    '''Downloads all the files corresponding to points within the given country from MERRA. Each file contains hourly wind speed data for the point
    for the given year and is saved into a folder named after the country.'''
    
    stime = time.time()
    logger.info('Initiating download process for {}.'.format(country))
    SW_coord = (min(europe_merra_points.loc[europe_merra_points.country==country].bounds.miny), min(europe_merra_points.loc[europe_merra_points.country==country].bounds.minx))
    NE_coord = (max(europe_merra_points.loc[europe_merra_points.country==country].bounds.maxy), max(europe_merra_points.loc[europe_merra_points.country==country,].bounds.maxx))
    download_path = os.path.join(SAF_directory,'data/MERRA',country)
    if not os.path.isdir(download_path):
        os.mkdir(download_path)
    download_subset(SW_coord,NE_coord,[year],download_path)
    logger.info(f'{country} files available after {time.time()-stime:.1f} seconds.')

establish_connection()

# Notice: Running the following cell will query the API and save the results to files

'''The following uses the multiprocessing library to download data for multiple countries simultaneously. 
The number of concurrent prcesses can be increased to decrease the ammount of time required to download all the data.
If you have slow wifi, try setting it to 4 or 5. If you download too fast, however, the data portal might ban you for a day.'''
countries = europe_merra_points['country'].unique()
concurrent_processes = 3
P = mp.Pool(concurrent_processes)
P.map(download_country_files,countries)
P.close()
P.join()
