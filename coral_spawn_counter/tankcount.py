#!/usr/bin/env/python3

# convert image count to tank count

# input: image count
# input: tank volume
# input: tank surface area

# initial guess: measurements of camera FOV

import numpy as np


image_count = 230

print()
print(f'image_count = {image_count} [count/image]')


## Surface Count assuming FOV of CSLICS:
# convert image counts to surface counts
# estimated tank surface area
rad_tank = 100.0/2 # cm^2 # actually measured the tanks this time
area_tank = np.pi * rad_tank**2

print(f'surface area of tank = {area_tank} [cm^2]')
# note: cslics surface area counts differ for different cslics!!
area_cslics = 2.3**2*(3/4) # cm^2 for cslics03 @ 15cm distance - had micro1 lens
# area_cslics = 2.35**2*(3/4) # cm2 for cslics01 @ 15.5 cm distance with micro2 lens
# area_cslics = 1.2**2*(3/4) # cm^2 prboably closer to this @ 10cm distance, cslics04

print(f'measured surface area of cslics (FOV at water level) = {area_cslics} [cm^2]')
nimage_to_tank_surface = area_tank / area_cslics
surface_count = image_count * nimage_to_tank_surface

print(f'surface count from CSLICS assuming known CSLICS FOV = {surface_count} [count]')
print()


## Surface Count given manual count
manual_count = 500000 # 500k
manual_count_lb = manual_count * 0.9
manual_count_ub = manual_count * 1.1


cslics_fov_est = image_count * area_tank / manual_count


print(f'Provided manual count = {manual_count}')
print(f'Manual count lower/upper bound = [{manual_count_lb}, {manual_count_ub}]')
print(f'Calibrated cslics_fov from manual count = {cslics_fov_est} [cm^2]')

# given constant cslics_fov, we can calculate the image_count

cslics_fov = 3.6128 # [cm2] # TODO set
print(f'setting cslics fov to {cslics_fov} [cm^2]')

cslics_surface_count = image_count * area_tank / cslics_fov

print(f'CSLICS surface count using FOV from manual count = {cslics_surface_count}')


print('done')