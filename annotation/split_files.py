#! /usr/bin/env python3

"""quick script to split files into train, val, test folders"""

import os
import shutil
import glob
import random

#if want to split train, val, test data
#directory has a images and labels subfolder
inpur_dir = '/home/java/Java/data/cslics_desktop_data/20241121_lobo'

train_ratio = 0.70
test_ratio = 0.15
valid_ratio = 0.15
def check_ratio(test_ratio,train_ratio,valid_ratio):
    if(test_ratio>1 or test_ratio<0): ValueError(test_ratio,f'test_ratio must be > 1 and test_ratio < 0, test_ratio={test_ratio}')
    if(train_ratio>1 or train_ratio<0): ValueError(train_ratio,f'train_ratio must be > 1 and train_ratio < 0, train_ratio={train_ratio}')
    if(valid_ratio>1 or valid_ratio<0): ValueError(valid_ratio,f'valid_ratio must be > 1 and valid_ratio < 0, valid_ratio={valid_ratio}')
    if not((train_ratio+test_ratio+valid_ratio)==1): ValueError("sum of train/val/test ratio must equal 1")
check_ratio(test_ratio,train_ratio,valid_ratio)

imagelist = glob.glob(os.path.join(inpur_dir,'images', '*.jpg'))
txtlist = glob.glob(os.path.join(inpur_dir, 'labels', '*.txt'))
txtlist.sort()
imagelist.sort()
imgno = len(txtlist) 
noleft = imgno

validimg, validtext, testimg, testtext = [], [], [], []

# function to seperate files into different lists randomly while retaining the same .txt and .jpg name in the specific type of list      
def seperate_files(number,newimglist,newtxtlist,oldimglist,oldtxtlist):
    for i in range(int(number)):
        r = random.randint(0, len(oldtxtlist) - 1)
        newimglist.append(oldimglist[r])
        newtxtlist.append(oldtxtlist[r])
        oldimglist.remove(oldimglist[r])
        oldtxtlist.remove(oldtxtlist[r])
    return oldimglist, oldtxtlist

imagelist, txtlist = seperate_files(imgno*valid_ratio,validimg,validtext,imagelist,txtlist)
imagelist, txtlist = seperate_files(imgno*test_ratio,testimg,testtext,imagelist,txtlist)

# function to preserve symlinks of src file, otherwise default to copy
def copy_link(src, dst):
    if os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, os.path.join(dst, os.path.basename(src)))
    else:
        shutil.copy(src, dst)
# function to make sure the directory is empty
def clean_dirctory(savepath):
    if os.path.isdir(savepath):
        shutil.rmtree(savepath)
    os.makedirs(savepath, exist_ok=True)
# function to move a list of files, by cleaning the path and copying and preserving symlinks
def move_file(filelist,savepathbase,savepathext):
    output_path = os.path.join(savepathbase, savepathext)
    #clean_dirctory(output_path)
    os.makedirs(output_path, exist_ok=True)
    for i, item in enumerate(filelist):
        copy_link(item, output_path)

move_file(txtlist,inpur_dir,'labels/train')
move_file(imagelist,inpur_dir,'images/train')
move_file(validtext,inpur_dir,'labels/val')
move_file(validimg,inpur_dir,'images/val')
move_file(testtext,inpur_dir,'labels/test')
move_file(testimg,inpur_dir,'images/test')

print("split complete")

import code
code.interact(local=dict(globals(), **locals()))