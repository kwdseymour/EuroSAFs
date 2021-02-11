
#!~/anaconda3/envs/GIS/bin/python
# coding: utf-8
 
class CoordinateError(Exception):
    '''Exception raised for errors in reading PV or wind data from files.

    Attributes:
        point -- coordinates which caused the error
        wind_or_PV -- the data set in which the point was not found
    '''
    def __init__(self, point, dataset):
        self.point = point
        self.dataset = dataset
        
    def __str__(self):
        return f'The given point {self.point} was not found in the {self.dataset} data set.'

class OptimizerError(Exception):
    '''Exception raised when the optimization fails.

    Attributes:
        point -- coordinates which caused the error
    '''
    def __init__(self, point):
        self.point = point
        
    def __str__(self):
        return f'Plant optimization for the given point {self.point} failed.'

