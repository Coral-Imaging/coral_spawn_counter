#! /usr/bin/env python3

"""
Annotated Region class, which has annotation and region properties
"""

from annotation.Region import Region

class AnnotationRegion(Region):
    # also potentially useful for the use of other metadata-driven annotations,
    # or occlusions, or other filtered attributes (none currently for CSLICS project)
    
    # types of annotation regions
    SHAPE_POLY = 'polygon' # more relevant for cgras than cslics
    SHAPE_RECT = 'rect'
    SHAPE_POINT = 'point'
    SHAPE_TYPES = [SHAPE_POINT, SHAPE_RECT, SHAPE_POLY]
    
    def __init__(self, class_name, shape_type, x_centre, y_centre, box_width, box_height):
        Region.__init__(self, class_name, shape_type, x_centre, y_centre, box_width, box_height)
        if shape_type in self.SHAPE_TYPES:
            self.shape_type = str(shape_type)
        else:
            TypeError('Unknown shape type passed to AnnotationRegion __init__()')
            print(f'shape_type = {shape_type}')
            self.shape_type = 'unknown'
            
            
    def print(self):
        Region.print(self)
        print(f'Shape type: {self.shape_type}')