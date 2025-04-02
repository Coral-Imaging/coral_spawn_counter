#! /usr/bin/env python3

"""
Region class, which corresponds to a polygon or bbox
"""

import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon

class Region:
    
    def __init__(self, 
                 class_name: str, 
                 shape_type: str, 
                 x_centre: float, 
                 y_centre: float, 
                 box_width: float, 
                 box_height: float):
        
        self.class_name = class_name
        self.shape_type = shape_type

        if not self.check_normalized(x_centre):
            ValueError(f'xc is not normalized: {x_centre}')
        if not self.check_normalized(y_centre):
            ValueError(f'yc is not normalized: {y_centre}')
        if not self.check_normalized(box_width):
            ValueError(f'w box_width is not normalized: {box_width}')
        if not self.check_normalized(box_height):
            ValueError(f'h box_height is not normalized: {box_height}')

        self.x_centre = x_centre
        self.y_centre = y_centre
        self.box_width = box_width
        self.box_height = box_height

        # x, y can be single, int/t, or an array of x's and y's
        # self.shape = self.make_shape(x, y)
    
    
    def check_normalized(self, x):
        """ check if variable is normalised """
        if x <= 1 and x > 0:
            return True
        else:
            return False


    def normalize_bbox(self, img_width, img_height, xc, yc, w, h):
        """normalize bounding box from pixels to 0-1

        Args:
            xc (_type_): _description_
            yc (_type_): _description_
            w (_type_): _description_
            h (_type_): _description
        """

        xc = float(xc) / float(img_width)
        yc = float(yc) / float(img_height)
        w = float(w) / float(img_width)
        h = float(h) / float(img_height)
        return [xc, yc, w, h]


    def unnormalize_bbox(self, img_width, img_height):
        xc = float(self.x_centre) * float(img_width)
        yc = float(self.y_centre) * float(img_height)
        w = float(self.box_width) * float(img_width)
        h = float(self.box_height) * float(img_height)
        return [xc, yc, w, h]

    # @staticmethod
    # def make_shape(x, y):
    #     """
    #     convert x, y values into shapely geometry object (point/polygon)
    #     NOTE: bbox is a type of polygon, accessed via shape.exterior.coords.xy
    #     """
        
    #     if type(x) is bool and type(y) is bool:
    #         shape = False
    #     elif type(x) is int and type(y) is int:
    #         shape = Point(x, y)
    #     else:
    #         # make sure x, y are same size
    #         x = np.array(x)
    #         y = np.array(y)
            
    #         if not x.shape == y.shape:
    #             TypeError('Inputs x, y are not the same shape')
    #         else:
    #             points = []
    #             i = 0
    #             while i < len(x):
    #                 # should have no negative x, y image coordinates
    #                 if x[i] < 0:
    #                     x[i] = 0
    #                 if y[i] < 0:
    #                     y[i] = 0
    #                 points.append(Point(int(x[i]), (y[i])))
    #                 i += 1
                    
    #         shape = Polygon(points)
            
    #     return shape
    
    
    def print(self):
        """ print Region """
        print('Region:')
        print('Class: ' + str(self.class_name))
        if self.shape_type:
            # print(f'Shape type: {self.shape.type}')
            print(f'Shape type: {self.shape_type}')
        else:
            print('shape is of unknown type')
        
        print('Coordinates:')
        print(f'x_centre: {self.x_centre}')
        print(f'y_centre: {self.y_centre}')
        print(f'box_width: {self.box_width}')
        print(f'box_height: {self.box_height}')
        # if type(self.shape) is Point:
        #     print(self.shape.bounds)
        # elif type(self.shape) is Polygon:
        #     print('polygon coordinates')
        #     print(self.shape.exterior.coords.xy)
        # else:
        #     print('unknown region shape type')
            
            
if __name__ == "__main__":
    
    print('Region.py')
    
    """ test shape input"""
    
    # test a yolo bbox
    pt_region = Region('egg', 'rect', 0.5, 0.6, 0.1, 0.2)
    pt_region.print()
    
    # test a polygon
    # all_x = (1, 2, 3, 4)
    # all_y = (3, 4, 5, 6)
    # poly_region = Region('egg', all_x, all_y)
    # poly_region.print()
    
    