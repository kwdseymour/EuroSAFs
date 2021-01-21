Master_Thesis

step no. | name                     | input data                                   | output data 
---------|--------------------------|----------------------------------------------|---------------------------------------------------------------------------------------------
   01    | 00_country_boundaries.py |- Country borders worldwide                   |- Country borders EU
         |                          |  './data/Countries_WGS84/Countries_WGS84.shp'|  './data/Countries_WGS84/Europe_WGS84.shp'
         |                          |- List of EU countries                        |- Evaluation Grid EU
         |                          |  './data/EU_EFTA_Countries.csv'              |  './data/Countries_WGS84/Europe_Evaluation_Grid.shp'
         |                          |                                              |- Evaluation Grid EU Coast
         |                          |                                              |  './data/Countries_WGS84/Coast_Evaluation_Grid.shp'
         |                          |                                              |- Evaluation Grid EU (coast only)
         |                          |                                              |  './data/Countries_WGS84/Europe_Coast_Evaluation_Grid.shp'
         |                          |                                              |- Evaluation Points EU
         |                          |                                              |  './data/Countries_WGS84/Europe_Evaluation_Points.shp'
         |                          |                                              |  './data/Countries_WGS84/Europe_Evaluation_Points.json'
         |                          |                                              |- Evaluation Points Coast
         |                          |                                              |  './data/Countries_WGS84/Coast_Evaluation_Points.shp'
         |                          |                                              |  './data/Countries_WGS84/Coast_Evaluation_Points.json'
---------|--------------------------|----------------------------------------------|----------------------------------------------------------------------------------------------
