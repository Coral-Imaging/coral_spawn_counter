#! /usr/bin/env python3

# grab random files from a set of folders/subfolders and copy them into a designated folder

import os
import glob
import shutil
import random
import code # for debugging

# NOTE for now, we do the dumb thing and just manually specify the folder(s)
# specify target directories (via glob) and image patterns of image files to find
# img_dir = '/media/dorian/subcslics23/cslics_subsurface_dataset/images/20231102_aant_tank3_cslics06/images/20231104'

random.seed(42)

def copy_random_files(img_dir, out_dir, img_pattern, num_images_get):
    # copy random number of files from one folder (img_dir) to another folder (out_dir)
    # img_pattern identifies the image files in the given folder
    # num_images_get is the number of images to copy over, selected via random.sample()
    
    os.makedirs(out_dir, exist_ok=True)
    print(f'cslics run: {img_dir}')
    # iterate over the relevant dates
    for d in dates:
        print(f'date = {d}')
        img_list = glob.glob(os.path.join(img_dir, d, img_pattern))
        print(f'img_dir = {img_dir}')
        print(f'len of img_list = {len(img_list)}')

        # randomly sort through list, grabbing images, copying them over
        img_select = random.sample(img_list,num_images_get)

        # copy images over
        print(f'copying {num_images_get} over to: {out_dir}')
        for i, img_name in enumerate(img_select):
            base_name = os.path.basename(img_name)
            shutil.copy(img_name, os.path.join(out_dir, base_name))
    return True

# TODO: clear out run folder before running this code, otherwise, will just add in/accumulate files in specified outdir

# choose date where cslics is definitely submerged
# also, output directory

# specify img pattern
img_pattern = '*/*clean.jpg'
# specify how many images to get 
num_images_get = 25 # only take 1% for training? ~15k images/day
# for quicker turn-around, we'll say 50/day

# 2024 november data (pdae)
run_list = ['202411_t5_pdae_100000000846a7ff']

# 2024 october data
# run_list = ['100000000029da9b',
#             '100000009c23b5af']
# 2023 data
# run_list = ['20231102_aant_tank3_cslics06',
#        '20231103_aten_tank4_cslics08',
#        '20231204_alor_tank3_cslics06',
#        '20231205_alor_tank4_cslics08']

for run in run_list:
    
    if run == '20231102_aant_tank3_cslics06':
        img_dir = '/media/dorian/DT4TB/cslics_2023_datasets/2023_Nov_Spawning/20231102_aant_tank3_cslics06/images'
        dates = ['20231105', '20231106', '20231107', '20231108', '20231109']
        out_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231102_aant_tank3_cslics06/images'

    elif run == '20231103_aten_tank4_cslics08':
        img_dir = '/media/dorian/DT4TB/cslics_2023_datasets/2023_Nov_Spawning/20231103_aten_tank4_cslics08/images'
        dates = ['20231106', '20231107', '20231108', '20231109']
        out_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231103_aten_tank4_cslics08/images'

    elif run == '20231204_alor_tank3_cslics06':
        img_dir = '/media/dorian/DT4TB/cslics_2023_datasets/2023_Dec_Spawning/20231204_alor_tank3_cslics06/images'
        dates = ['20231206', '20231207', '20231208', '20231209']
        out_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231204_alor_tank3_cslics06/images'

    elif run == '20231205_alor_tank4_cslics08':
        img_dir = '/media/dorian/DT4TB/cslics_2023_datasets/2023_Dec_Spawning/20231205_alor_tank4_cslics08/images'
        dates = ['20231207', '20231208', '20231209', '20231210', '20231211','20231212']
        out_dir = '/home/dorian/Data/cslics_2023_subsurface_dataset/runs/20231205_alor_tank4_cslics08/images'

    elif run == '100000000029da9b':
        img_dir = '/media/dorian/T2D/cslics_2024_october/20241023_spawning/100000000029da9b'
        dates = ['2024-10-24', '2024-10-25','2024-10-26','2024-10-27','2024-10-28']
        out_dir = '/home/dorian/Data/cslics_datasets/cslics_2024_october_subsurface_dataset/100000000029da9b/images'
        
    elif run == '100000009c23b5af':
        img_dir = '/media/dorian/T2D/cslics_2024_october/20241023_spawning/100000009c23b5af'
        dates = ['2024-10-25','2024-10-26','2024-10-27','2024-10-28']
        out_dir = '/home/dorian/Data/cslics_datasets/cslics_2024_october_subsurface_dataset/100000009c23b5af/images'
    
    elif run == '202411_t4_pdae_100000001ab0438d':
        img_dir = '/media/dtsai/CSLICSNov24/cslics_november_2024/100000001ab0438d'
        dates = ['2024-11-23','2024-11-24','2024-11-25','2024-11-26', '2024-11-27','2024-11-28']
        out_dir = '/home/dtsai/Data/cslics_datasets/cslics_2024_november_subsurface_dataset/100000001ab0438d/images'
        
    elif run == '202411_t5_pdae_100000000846a7ff':
        img_dir = '/media/dtsai/CSLICSNov24/cslics_november_2024/100000000846a7ff'
        dates = ['2024-11-20', '2024-11-21', '2024-11-22', '2024-11-23','2024-11-24']
        out_dir = '/home/dtsai/Data/cslics_datasets/cslics_2024_november_subsurface_dataset/100000000846a7ff/images'
        
    else:
        print('specify run')

    copy_random_files(img_dir, out_dir, img_pattern, num_images_get)
    # end run loop
    
print('done')

# code.interact(local=dict(globals(),**locals()))

