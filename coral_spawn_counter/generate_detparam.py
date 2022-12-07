#! /usr/bin/env python3

# generate detection parameters json for given image settings

import json
import os

# save file information
save_path = '/media/agkelpie/cslics_ssd/2022_NovSpawning/20221112_AMaggieTenuis/cslics04/metadata'
save_file = 'circ_det_param.json'

# dictionary to save:
# detection parameters

circle_det_param = {'blur': 9,
                     'dp': 2.5,
                     'minDist': 45,
                     'param1': 45,
                     'param2': 0.5,
                     'maxRadius': 80,
                     'minRadius': 45}

# save det param to file
save_det_param_file = os.path.join(save_path, save_file)
with open(save_det_param_file, 'w') as f:
    json.dump(circle_det_param, f)

# detection parameters for far focus cslics: cslics2:
# det_param_far = {'blur': 5,
#                 'dp': 1,
#                 'minDist': 25,
#                 'param1': 100,
#                 'param2': 0.1,
#                 'maxRadius': 40,
#                 'minRadius': 15}

# det_param_med_cslics01 = {'blur': 7,
#                 'dp': 1,
#                 'minDist': 25,
#                 'param1': 75,
#                 'param2': 0.3,
#                 'maxRadius': 70,
#                 'minRadius': 20}

# # detection parameters for near focus cslics: cslics04
# det_param_close_cslics04 = {'blur': 9,
#                 'dp': 2.5,
#                 'minDist': 50,
#                 'param1': 50,
#                 'param2': 0.5,
#                 'maxRadius': 80,
#                 'minRadius': 50}

# det_param_close_cslics03 = {'blur': 9,
#                 'dp': 2.5,
#                 'minDist': 50,
#                 'param1': 50,
#                 'param2': 0.5,
#                 'maxRadius': 80,
#                 'minRadius': 50}

# # parameters for HOUGH_GRADIENT (not HOUGH_GRADIENT_ALT)
# # det_param_close_cslics03 = {'blur': 5,
# #                 'dp': 1.35,
# #                 'minDist': 50,
# #                 'param1': 75,
# #                 'param2': 20,
# #                 'maxRadius': 80,
# #                 'minRadius': 50}

# det_param_wide = {'blur': 3,
#                 'dp': 2.5,
#                 'minDist': 5,
#                 'param1': 50,
#                 'param2': 0.5,
#                 'maxRadius': 12,
#                 'minRadius': 5}  # no detection parameters for wide FOV yet

# host_det_param = {"cslics01": det_param_med_cslics01,
#               "cslics02": det_param_far,
#               "cslics03": det_param_close_cslics03,
#               "cslics04": det_param_close_cslics04,
#               "cslics06": det_param_wide,
#               "cslics07": det_param_wide,
#               "cslicsdt": det_param_wide
#              }

# wide_lens = ['cslics06', 'cslics07']