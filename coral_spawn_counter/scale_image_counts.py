#! /usr/bin/env python3

def get_hyper_focal_dist(f, c, n):
    return f + f**2 / (c * n)

def scale_by_focus_volume():
    # a physics-based approach to solving the calibration problem    
    # issue with this approach is that it requires very careful calibration, which can/may vary with different cslics
    # this approach puts significant onus on careful calibration for every single unit
    # and still relies on a somewhat nebulous variable "c", the circle of confusion

    width_pix = 4056 # pixels
    height_pix = 3040 # pixels
    pix_size = 1.55 / 1000 # um -> mm, pixel size
    sensor_width = width_pix * pix_size # mm
    sensor_height = height_pix * pix_size # mm
    f = 12 # mm, focal length
    aperture = 2.8 # f-stop number of the lens
    
    c = 0.2 # mm, circle of confusion, def 0.1, increase to 0.2 to double (linear) the sample volume
    
    hyp_dist = get_hyper_focal_dist(f, c, aperture) # hyper-focal distance = max depth of field of camera
    focus_dist = 75 #mm focusing distance, practically the working distance of the camera
    # NOTE: focus distance was kept to ~ the same in the CSLICS 2023, but may differ between CSLICS (see CSLICS tank setup notes)
    dof_far = (hyp_dist * focus_dist) / (hyp_dist - (focus_dist - f))
    dof_near = (hyp_dist * focus_dist) / (hyp_dist + (focus_dist - f))
    dof_diff = abs(dof_far - dof_near) # mm
    print(f'DoF diff = {dof_diff} mm')

    work_dist = focus_dist # mm, working distance
    # 1.33 for refraction through water, lensing effect
    hfov = work_dist * sensor_height / (1.33 * f) # mm, horizontal field-of-view
    vfov = work_dist * sensor_width / (1.33 * f) # mm, vertical field-of-view
    print(f'horizontal FOV = {hfov}')
    print(f'vertical FOV = {vfov}')

    area_cslics = hfov * vfov # mm, area of cslics
    print(f'area_cslics = {area_cslics} mm^2')

    # we can approximate the frustum as a rectangular prism, since the angular FOV is not that wide
    focus_volume = area_cslics * dof_diff # mm^3
    print(f'focus volume = {focus_volume} mm^3')
    print(f'focus volume = {focus_volume/1000} mL')

    volume_image = focus_volume / 1000 # Ml # VERY MUCH AN APPROXIMATION - TODO FIGURE OUT THE MORE PRECISE METHOD
    volume_tank = 475 * 1000 # 500 L = 500000 ml
    
    scale_factor = volume_tank / volume_image # thus, how many cslics images will fill the whole volume of the tank
    print(f'scale factor = {scale_factor}')

    return scale_factor

scale_by_focus_volume()