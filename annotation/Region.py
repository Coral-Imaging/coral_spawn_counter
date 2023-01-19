#! /usr/bin/env python3

"""
Region class, which corresponds to a polygon or bbox
"""

import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon

class Region:
    
    def __init__(self, class_name: str, x, y):
        self.class_name = class_name
        
        # x, y can be single, int/t, or an array of x's and y's
        self.shape = self.make_shape(x, y)
        
    @staticmethod
    def make_shape(x, y):
        """
        convert x, y values into shapely geometry object (point/polygon)
        NOTE: bbox is a type of polygon, accessed via shape.exterior.coords.xy
        """
        
        if type(x) is bool and type(y) is bool:
            shape = False
        elif type(x) is int and type(y) is int:
            shape = Point(x, y)
        else:
            # make sure x, y are same size
            x = np.array(x)
            y = np.array(y)
            
            if not x.shape == y.shape:
                TypeError('Inputs x, y are not the same shape')
            else:
                points = []
                i = 0
                while i < len(x):
                    # should have no negative x, y image coordinates
                    if x[i] < 0:
                        x[i] = 0
                    if y[i] < 0:
                        y[i] = 0
                    points.append(Point(int(x[i]), (y[i])))
                    i += 1
                    
            shape = Polygon(points)
            
        return shape
    
    
    def print(self):
        """ print Region """
        print('Region:')
        print('Class: ' + str(self.class_name))
        if self.shape:
            print(f'Shape type: {self.shape.type}')
        else:
            print('shape is of unknown type')
        
        print('Coordinates:')
        if type(self.shape) is Point:
            print(self.shape.bounds)
        elif type(self.shape) is Polygon:
            print('polygon coordinates')
            print(self.shape.exterior.coords.xy)
        else:
            print('unknown region shape type')
            
            
if __name__ == "__main__":
    
    print('Region.py')
    
    """ test shape input"""
    
    # test a point
    pt_region = Region('point_a', 1, 4)
    pt_region.print()
    
    # test a polygon
    all_x = (1, 2, 3, 4)
    all_y = (3, 4, 5, 6)
    poly_region = Region('egg', all_x, all_y)
    poly_region.print()
    
    