# Description

This repository can be used to evaluate the minimum cost of fuel producation at sites throughout Europe using a variety of energy resource, cost, efficiency, and other collected data. The minimum cost is obtained via a plant optimization process in which the cost is minimized by selecting optimal plant component sizes and hourly operation (energy & mass flows). 

# Setup
The [environments.yml](https://github.com/kwdseymour/EuroSAFs/blob/master/environment.yml) file can be used to set up a working Python environment for the scripts in the repository. A Gurobi license must be obtained in order to run the optimization process.

# [Data Preparation](https://github.com/kwdseymour/EuroSAFs/tree/master/scripts/data_preparation)
## [country_boundaries.py](https://github.com/kwdseymour/EuroSAFs/blob/master/scripts/data_preparation/country_boundaries.py)
Divide each country into a grid of evaluation nodes.
#### Inputs
A [shapefile](https://github.com/kwdseymour/EuroSAFs/blob/master/data/Countries_WGS84/Countries_WGS84.shp) with countaining country borders is required for the script. It is available in the data folder: [data/Countries_WGS84](https://github.com/kwdseymour/EuroSAFs/tree/master/data/Countries_WGS84).
#### Ouputs
Found in the [data/Countries_WGS84/processed](https://github.com/kwdseymour/EuroSAFs/tree/master/data/Countries_WGS84/processed) folder, the resulting files contain the country grids in various forms. These determine the locations in which wind speed and PV output data are collected and the plant optimization takes place.

## [MERRA_download.py](https://github.com/kwdseymour/EuroSAFs/blob/master/scripts/data_preparation/MERRA_download.py)
Download wind speed data at each evaluation node from NASA's MERRA-2 reanalysis data API. An Earthdata account is required to download data and the credentials should be stored in the [config file](https://github.com/kwdseymour/EuroSAFs/blob/master/scripts/config_template.json). An account can be created [here](https://urs.earthdata.nasa.gov/).
#### Outputs
For each country given, the nearest MERRA point is identified and the hourly data for all points in each day of 2016 is downloaded to a file in a folder named according to the country.




![alt text](https://github.com/kwdseymour/EuroSAFs/blob/master/gfx/LCOF_combined.png)
