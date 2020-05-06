#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:06:32 2020

@author: ryan
"""
# Code from Image Analysis
def splitAVI(dates,src='data/avi/',dest='data/img/',status=False):
    import cv2
    import os
    
    
    for date in dates:
        if not os.path.exists(f"{src}{date}"):
            print(f"{src}{date} directory does not exist")
            return
        
        if not os.path.exists(f'{dest}{date}'):
            os.makedirs(f'{dest}{date}')
        
        for vid in os.listdir(f'{src}{date}/'):
            vidcap = cv2.VideoCapture(f'{src}{date}/{vid}')
            success,image = vidcap.read()
            count = 0
            
            fname = vid.split('.avi')[0]
            
            if not os.path.exists(f'{dest}{date}/{fname}'):
                os.makedirs(f'{dest}{date}/{fname}')
            
            while success:
              cv2.imwrite(f'{dest}{date}/{fname}/{count}.png', image)   
              success,image = vidcap.read()
              #print ('Read a new frame: ', success)
              count += 1
              
            
            cv2.VideoCapture.release(vidcap)
            
        if status:
            print(f"Processed {date}")

# More code from Image Analysis
def getList(start,end):
    """
    Strictly Ymd format no spaces or slashes
    """
    
    import datetime
    
    start = datetime.datetime.strptime(str(start),'%Y%m%d')
    end = datetime.datetime.strptime(str(end),'%Y%m%d')
    
    dateList = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days + 1)]
    
    sList = [datetime.datetime.strftime(x,'%Y%m%d') for x in dateList]
    
    return sList


#Splitting
dateList = getList('20200201','20200201')
dateList2 = getList('20200422','20200422')
splitAVI(dateList,status=True)
splitAVI(dateList2, status=True)

# code written by Trajectories

def saveFITS(img, name):
    from astropy.io import fits
    hdu = fits.PrimaryHDU(img)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name + '.fits')
    
def stack(dates,src='data/',status=False, save_fits = False):
    import cv2
    import os
    import glob
    import numpy as np
    
    for date in dates:
        
        obsv = src + 'avi/' + date + '/'
        if not os.path.exists(obsv):
            print(f"{obsv} directory does not exist, split avi first")
            return
        
        # create destination for stacked images
        if not os.path.exists(src + 'img/' + date + '/stacked/'):
            os.makedirs(src + 'img/' + date + '/stacked/')
        
        images_out = []
        
        # look at each observation for the particular date
        for vid in glob.glob(src + 'img/' + date + '/' + 'ev*/'):
            
            # create list of strings of image names
            image_list = glob.glob(vid + '*.png')
            # read in first image using cv2 in grayscale and convert to float64
            image = np.full_like(cv2.imread(image_list[0], cv2.IMREAD_GRAYSCALE), 0)
            image = image - np.full_like(image, np.median(image))
#            image = np.full_like(image1, 0)
            # iterate through list of pngs
            for png in range(0, len(image_list)):
                # load image and convert to float64
                img = np.float64(cv2.imread(image_list[png], cv2.IMREAD_GRAYSCALE))
#                img = np.float64(cv2.imread(image_list[png], cv2.IMREAD_GRAYSCALE)) - np.float64(cv2.imread(image_list[0], cv2.IMREAD_GRAYSCALE))
                img = img - np.full_like(img, np.median(img))
                # add the new image to the intial image and redifine the initial image
                image = image + img
#                image = image - np.full_like(image, np.median(image))
                print('added')
            image = image/len(image_list)
            if save_fits:
                saveFITS(image, 'data/img/' + date + '/stacked/' + vid.split('/')[-2])
            # append the stacked image to a list
            images_out.append(image)
        return images_out

obs = stack(dateList2, save_fits = True)

def plotimage(image, vmin, vmax, fig_name = None):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    if fig_name is not None:
        plt.figure(fig_name)
        plt.imshow(image, cmap='gray', norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax))
        plt.colorbar()
        
    else:
        plt.figure(fig_name)
        plt.imshow(image, cmap='gray', norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax))
        plt.colorbar()
        
for i in obs:
    plotimage(i, vmin = 1000, vmax = 5000)
    


# plotting subset of one image to get profile of a star.
import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(435,450), obs[0][322,435:450])
