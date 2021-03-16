# Description

This repository provides a tool to evaluate the minimum cost of fuel producation at sites throughout Europe using a botoom-up plant modeling approach in which technical and financial assumptions are combined with wind & solar resource data. The minimum cost is obtained via an optimization problem in which the cheapest plant configuration is determined by selecting component sizes and defining hourly operation (energy & mass flows). 

# Setup
The [environments.yml](https://github.com/kwdseymour/EuroSAFs/blob/master/environment.yml) file can be used to set up a working Python environment for the scripts in the repository. A Gurobi license must be obtained in order to run the optimization process.

# Core functionality
At the heart of the repository is the [plant_optimization](https://github.com/kwdseymour/EuroSAFs/tree/master/scripts/optimization/plant_optimization) module, which contains the [plant optimizer](https://github.com/kwdseymour/EuroSAFs/blob/master/scripts/optimization/plant_optimization/plant_optimizer.py) classes and function. A simple use case is as follows:

    import plant_optimization as plop
    
    # Define the evaluation country
    country = "Austria"
    
    # Define the plant evaluation location coordinates
    location = (47.0, 9.375)
    
    # Initialize a Site object, which extracts and contains the hourly wind & PV power output for the given location
    site = plop.Site(location,country,offshore=False)
    
    # Initialize a Plant object, which extracts and contains:
    # - the plant parameters (costs, efficiencies, operation constraints, etc.) according to the provided year
    # - the given Site object
    plant = plop.Plant(site,year=2020)
    
    # Run the plant optimizer, which saves the solution to the Plant object
    pop.optimize_plant(plant)


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
