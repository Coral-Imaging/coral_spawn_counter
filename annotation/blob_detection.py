#! /usr/bin/env python3

"""blob_detection.py
detect blobs for sub-surface annotation
"""


import os
import cv2 as cv
import numpy as np
import glob
import random as rng
import matplotlib.pyplot as plt

# image dir
# read in images
# binarize image
# set thresholds
# morphological operations
# find blobs
# count blobs

def getchildren(contours, hierarchy):
    # gets list of children for each contour based on hierarchy
    # follows similar for loop logic from _hierarchicalmoments, so
    # TODO use _getchildren to cut redundant code in _hierarchicalmoments

    children = [None]*len(contours)
    for i in range(len(contours)):
        inext = hierarchy[i, 0]
        ichild = hierarchy[i, 2]
        if not (ichild == -1):
            # children exist
            ichild = [ichild]
            otherkids = [k for k in range(i + 1, len(contours))
                            if ((k < inext) and (inext > 0))]
            if not len(otherkids) == 0:
                ichild.extend(list(set(otherkids) - set(ichild)))
            children[i] = ichild
        else:
            # else no children
            children[i] = [-1]
    return children
    
def hierarchicalmoments(contours, hierarchy, mu):
    # for moments in a hierarchy, for any pq moment of a blob ignoring its
    # children you simply subtract the pq moment of each of its children.
    # That gives you the “proper” pq moment for the blob, which you then
    # use to compute area, centroid etc. for each contour
    #   find all children (row i to hierarchy[0,i,0]-1, if same then no
    #   children)
    #   recompute all moments

    # to deliver all the children of i'th contour:
    # first index identifies the row that the next contour at the same
    # hierarchy level starts
    # therefore, to grab all children for given contour, grab all rows
    # up to i-1 of the first row value
    # can only have one parent, so just take the last (4th) column

    # hierarchy order: [Next, Previous, First_Child, Parent]
    # for i in range(len(contours)):
    #    print(i, hierarchy[0,i,:])
    #    0 [ 5 -1  1 -1]
    #    1 [ 4 -1  2  0]
    #    2 [ 3 -1 -1  1]
    #    3 [-1  2 -1  1]
    #    4 [-1  1 -1  0]
    #    5 [ 8  0  6 -1]
    #    6 [ 7 -1 -1  5]
    #    7 [-1  6 -1  5]
    #    8 [-1  5  9 -1]
    #    9 [-1 -1 -1  8]

    mh = mu
    for i in range(len(contours)):  # for each contour
        inext = hierarchy[i, 0]
        ichild = hierarchy[i, 2]
        if not (ichild == -1):  # then children exist
            ichild = [ichild]  # make first child a list
            # find other children who are less than NEXT in the hierarchy
            # and greater than -1,
            otherkids = [k for k in range(i + 1, len(contours)) if
                            ((k < inext) and (inext > 0))]
            if not len(otherkids) == 0:
                ichild.extend(list(set(otherkids) - set(ichild)))

            for j in range(ichild[0], ichild[-1]+1):  # for each child
                # all moments that need to be computed
                # subtract them from the parent moment
                # mh[i]['m00'] = mh[i]['m00'] - mu[j]['m00'] ...

                # do a dictionary comprehension:
                mh[i] = {key: mh[i][key] -
                            mu[j].get(key, 0) for key in mh[i]}
        # else:
            # no change to mh, because contour i has no children

    return mh


def computecentroids(contours, moments):
    mf = moments
    mc = [(mf[i]['m10'] / (mf[i]['m00']), mf[i]['m01'] / (mf[i]['m00']))
            for i in range(len(contours))]
    return mc

def computearea(colntours, moments):
    return [moments[i]['m00'] for i in range(len(contours))]  
    

def drawBlobs(image,
              contours,
              hierarchy,
              uc,
              vc,
            drawing=None,
            icont=None,
            color=None,
            contourthickness=cv.FILLED,
            textthickness=2):
        """
        Draw the blob contour

        :param image: [description]
        :type image: [type]
        :param drawing: [description], defaults to None
        :type drawing: [type], optional
        :param icont: [description], defaults to None
        :type icont: [type], optional
        :param color: [description], defaults to None
        :type color: [type], optional
        :param contourthickness: [description], defaults to cv.FILLED
        :type contourthickness: [type], optional
        :param textthickness: [description], defaults to 2
        :type textthickness: int, optional
        :return: [description]
        :rtype: [type]
        """
        # draw contours of blobs
        # contours - the contour list
        # icont - the index of the contour(s) to plot
        # drawing - the image to draw the contours on
        # colors - the colors for the icont contours to be plotted (3-tuple)
        # return - updated drawing

        # TODO split this up into drawBlobs and drawCentroids methods

        # image = Image(image)
        # image = self.__class__(image)  # assuming self is Image class
        # @# assume image is Image class

        if drawing is None:
            drawing = np.zeros(
                (image.shape[0], image.shape[1], 3), dtype=np.uint8)

        if icont is None:
            icont = np.arange(0, len(contours))
        else:
            icont = np.array(icont, ndmin=1, copy=True)

        if color is None:
            # make colors a list of 3-tuples of random colors
            color = [None]*len(icont)

            for i in range(len(icont)):
                color[i] = (rng.randint(0, 256),
                            rng.randint(0, 256),
                            rng.randint(0, 256))
                # contourcolors[i] = np.round(colors[i]/2)
            # TODO make a color option, specified through text,
            # as all of a certain color (default white)

        # make contour colours slightly different but similar to the text color
        # (slightly dimmer)?
        cc = [np.uint8(np.array(color[i])/2) for i in range(len(icont))]
        contourcolors = [(int(cc[i][0]), int(cc[i][1]), int(cc[i][2]))
                         for i in range(len(icont))]

        # TODO check contours, icont, colors, etc are valid
        hierarchy = np.expand_dims(hierarchy, axis=0)
        # done because we squeezed hierarchy from a (1,M,4) to an (M,4) earlier

        for i in icont:
            # TODO figure out how to draw alpha/transparencies?
            cv.drawContours(drawing,
                            contours,
                            icont[i],
                            contourcolors[i],
                            thickness=contourthickness,
                            lineType=cv.LINE_8,
                            hierarchy=hierarchy)

        for i in icont:
            ic = icont[i]
            cv.putText(drawing,
                       str(ic),
                       (int(uc[ic]), int(vc[ic])),
                       fontFace=cv.FONT_HERSHEY_SIMPLEX,
                       fontScale=1,
                       color=color[i],
                       thickness=textthickness)

        return drawing
    
    

img_dir = '/home/dorian/Data/cslics_2022_datasets/subsurface_data/20221113_amtenuis_cslics04/images_subset'
img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
# img_dir = '/home/dorian/Code/turtles/turtle_datasets/job10_mini/frames_0_200'
# img_list = sorted(glob.glob(os.path.join(img_dir, '*.PNG')))

save_dir = 'output'
os.makedirs(save_dir, exist_ok=True)


max_img = 10
for i, img_name in enumerate(img_list):
    if i >= max_img:
        print('hit max img')
        break
    
    img = cv.imread(img_name, 0) # BGR

    img_height, img_width = img.shape
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # RGB
    # desired_width = 1280
    # desired_height = int(np.ceil(img_height / img_width * desired_width))
    # img = cv.resize(img, (desired_width, desired_height))
    
    # TODO image processing/smoothing?
    ksize = 61 # very high due to large noise and large scale features
    img = cv.GaussianBlur(img, (ksize, ksize), 0)
    
            
    # import code
    # code.interact(local=dict(globals(), **locals()))
    
    # CANNY EDGE DETECTION
    canny = cv.Canny(img, 3, 5, L2gradient=True)
    
    
    CANNY_GUI = False
    if CANNY_GUI:
        # cre`ate binary image (findcontours will treat anything > 1 as 1)
        # use canny edge filter 
        # https://docs.opencv.org/4.6.0/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de
        low_threshold = 255/3 # first threshold for hysteresis procedure, smallest used for edge linking
        high_threshold = 5 # second, largest value used to find intiial segments for strong edges
        apertureSize = 5
        
        canny = cv.Canny(img, 85, 255, 5)
        
        def callback(x):
            print(x)
            
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        cv.createTrackbar('L', 'image', 0, 255, callback)
        cv.createTrackbar('U', 'image', 0, 255, callback)
        # cv.createTrackbar('A', 'image', 0, 300, callback)
        while (1):
            np_horz_concat = np.concatenate((img, canny), axis=1) # to display image side-by-side
            cv.imshow('image', np_horz_concat)
            k = cv.waitKey(1) & 0xFF
            if k == 27: # escape key
                break
            l = cv.getTrackbarPos('L', 'image')
            u = cv.getTrackbarPos('U', 'image')
            # a = cv.getTrackbarPos('A', 'image')
            canny = cv.Canny(img, l, u, L2gradient=True)
            # img_e = cv.Can`ny(img, low_threshold, high_threshold, apertureSize, L2gradient=True)
    
    # morphological operations
    k = 21
    kernel = np.ones((k,k), np.uint8)
    canny = cv.dilate(canny, kernel, iterations = 1)
    canny = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
    canny = cv.morphologyEx(canny, cv.MORPH_OPEN, kernel)
    
    save_img_name = os.path.basename(img_name)[:-4] + '_blob.jpg'
    
    plt.imsave(os.path.join(save_dir, save_img_name), canny)
    
    
    # show the image
    # fig, ax = plt.subplots()
    # ax.imshow(canny)
    
    # fig2, ax2 = plt.subplots()
    # ax2.imshow(img_e)
    
    # plt.show()
    
    # cv.simpleblobdetector too simple, cannot get pixel values/locations of
    # # blobs themselves, so instead find contours
    contours, hierarchy = cv.findContours(canny, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    
    hierarchy = hierarchy[0,:,:] # drop the first singleton dimension
    parent = hierarchy[:, 2]
    children = getchildren(contours, hierarchy)
    
    
    # # get moments as a dictionary for each contour
    mu = [cv.moments(contours[i])
            for i in range(len(contours))]
    
    moments = hierarchicalmoments(contours, hierarchy, mu)
    
    # # get mass centers/centroids:
    mc = np.array(computecentroids(contours, moments))
    uc = mc[:, 0]
    vc = mc[:, 1]
    
    drawing = drawBlobs(canny, contours, hierarchy, uc, vc)
    
    fig, ax = plt.subplots()
    ax.imshow(drawing) 
    plt.show()
    
    # TODO reject too-small and too weird blobs
    # TODO write to xml file to upload blob annotations