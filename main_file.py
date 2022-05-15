# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 13:07:53 2022

@author: juliesi
"""

import scipy.io as scio
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt 
import glob  # used to return all file paths that match a specific pattern, eks. set of file names
import pandas as pd  
import re # let you check if a particular string matches a given regular expression (or if a given regular expression matches a particular string
import os

"""
This code takes an all-sky image as input and uses Otsu's threshold 
segmentation to find the threshold of the northern light. 


GNSS data gives us information about the azimutal angle and the elevetion, as 
well as the scintilattion. This is used to plot the scintillations over the 
threshold image of the northern light.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

There are 6(six) places you need to change the date in order to look at a 
nother day;

    all_ims = [im for im in all_ims if im.__contains__("20190104")] 
    
    calib = scio.readsav('asi/nya6_20190104_5577_cal.dat')
 
    files = glob.glob('20190104_NYA_REDOBS_gps.txt', recursive=True) 
    
    txt_path = open('C:/Users/juliesi/Documents/Master/Programmering/asi/txt data fra koden/04012019_grad7', 'w+')

    plt.title(ALTSÅ I TITTELEN PÅ PLOTTENE)

and 
    plt.savefig(pic_path+hhmm+'__04012019.png')
"""


### Path for importing the all sky images, NYA
ASI_path = 'asi/'
# ASI_img  = glob.glob(ASI_path + 'nya6_*_5577_cal.png')    # Looking at green Aurora, 557.7nm

all_ims = glob.glob(ASI_path + "*.png")
all_ims = [im for im in all_ims if im.__contains__("20190208")] ### Change the date to look at another day
### splits up the file name of ASI and gives only the time HHMMSS, 
### and give total minutes
img_minutes = [os.path.basename(im_name).split("_")[2] for im_name in all_ims]
img_minutes =[int(im_name[:2])*60 + int(im_name[2:4]) + int(im_name[4:6])/60 for im_name in img_minutes]
img_minutes = np.array(img_minutes)

### Path for importing the calibration data (ASI), NYA 
calib = scio.readsav('asi/nya6_20190208_5577_cal.dat')      ### Change the date to look at a nother day
# print(calib.keys()) 

Azimuth   = calib['gazms']      # Azimuthal angle from ASI calibration data
Elevation = calib['elevs']      # Elevation from ASI calibration data






"""
Reading gps data:
    ser på datoen 04.01.2019 (4. ian 2019), 11-12 UT, NYÅ
"""

# define which files I want to use
files = glob.glob('20190208_NYA_REDOBS_gps.txt', recursive=True) # choose the ending I want the file to have, I read in all the ones from gps in NYÅ

# startdate
gps_tow_start=datetime.datetime(1980,1,6,0,0,0) 

# reading gps scint
for i in range(len(files)):
    gps_temp = pd.read_csv(files[i], delimiter=',', header=[12],parse_dates=['HHMM'], skipinitialspace=True)
    week=pd.read_csv(files[i],delimiter=' ', header=None, skiprows = 16, nrows=1, skipinitialspace=True)
    week.replace(',','', regex=True, inplace=True)
    gps_temp.columns = gps_temp.columns.str.replace(' ', '')

    gps_temp['week'] = int(week.loc[0,2])
    gps_temp['SatSys'] = week.loc[0,5]
    DateDay = re.findall(r'\d+', files[i])[0]
    gps_temp['TOWDate'] = gps_tow_start + pd.to_timedelta(gps_temp.week*7, unit='d') + pd.to_timedelta(gps_temp.GPSTOW, unit='s')
   
gps = gps_temp




txt_path = open('C:/Users/juliesi/Documents/Master/Programmering/asi/txt data fra koden/10122019_ori5', 'w+')

pic_path = 'C:/Users/juliesi/Documents/Master/Programmering/asi/bilder fra koden/'


### size of the markes of scintillation indices in the legend 
s_square = [5, 12, 30, 52, 70]  


for hhmm in gps['HHMM'].unique(): 
     
   
    # Splits up the HHMM in GPS data, and gives out total minutes
    gps_minutes = int(hhmm[:2])*60 + int(hhmm[2:])
    
    
    # Finds the closest all sky imgage for the gps time
    index = np.argmin(np.abs(img_minutes - gps_minutes))
    
    # If the time-step between the image and gps is larger than 2 min: continue
    if np.abs(img_minutes[index] - gps_minutes) > 2:
        continue 
    
   
   
    ### Otsu's threshold segmentation
    
    # Loading the images
    img = cv2.imread(all_ims[index], cv2.IMREAD_GRAYSCALE)
    img = img[70:380, 70:380] #cropping the image   
    
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((2,2),np.uint8)

    #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(closing,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)

    # Threshold
    ret, sure_fg = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    

   


    ### Gradients in picture
    
    # compute the gradients along the x and y axis
    gX = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    gY = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)


 
    
    # # combine the gradient representations into a single image
    # combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
    
    # compute the gradient magnitude and orientation
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
    
    
    
    
    
    ### Here comes the scintilation dots
    
    #scint = gps.loc[gps['HHMM'] == hhmm, '60SecSigma']
    
    # Get all scintillation data for your current time step
    gps_now= gps.loc[gps["HHMM"] == hhmm, :]
    
    for i, row in gps_now.iterrows():
    
        diff =  np.abs(Azimuth - row['Az']) + np.abs(Elevation - row['Elv'])
    
        loc = np.unravel_index(np.nanargmin(diff), diff.shape) 

        # plot the different scintillation sizes
    
            
        if row['60SecSigma'] >= 0.2:
            plt.scatter(loc[0], loc[1], marker='.', s=row['60SecSigma']*450, color='dodgerblue')
            
        if row['60SecSigma'] < 0.1:
            plt.scatter(loc[0], loc[1], marker='.', s=row['60SecSigma']*300, color='gold')
            
        if  row['60SecSigma'] >= 0.1 and row['60SecSigma'] < 0.15:
            plt.scatter(loc[0], loc[1], marker='.', s=row['60SecSigma']*350, color='violet')
            
        if row['60SecSigma'] >= 0.15 and row['60SecSigma'] < 0.2:
            plt.scatter(loc[0], loc[1], marker='.', s=row['60SecSigma']*400, color='limegreen')
            
            txt_path.write(str(hhmm) + ' ')# + str(loc[0]) + ' ' + str(loc[1]) + ' ' + str(row['60SecSigma']) + ' ')
            
            txt_path.write(str(loc[0]) + ' ')
            txt_path.write(str(loc[1]) + ' ')
          
            txt_path.write(str(row['60SecSigma']) + ' ')
            
            try:
                txt_path.write(str(gX[loc[0]]) + ' ')
            except Exception:
                pass
            try:
                txt_path.write(str(gY[loc[1]]) + ' ')
            except Exception:
                pass
        
        
        
        
            try:
                txt_path.write(str(magnitude[loc]) + ' ')
            except Exception:
                pass
        
            try:
                txt_path.write(str(orientation[loc]) + ' ')
            except Exception:
                pass
            
        
        
        
            # try:
            #     txt_path.write(str(combined[loc]) + ' ')
            # except Exception:
            #     pass
            
            txt_path.write('\n')
            txt_path.write('\n')
            
            
            
            print('time: ', hhmm)
            print('location: ', loc)
            print('scint: ', row['60SecSigma'])
                
            #print('gX = ', gX[loc[0]])
            #print('gY = ', gY[loc[1]])   
            # try:
            #     print('combined', combined[loc])
            # except Exception:
            #     pass
            
            print('')
            print('')
            
            
            
            legend_elements = [plt.scatter([], [], marker='o', color='black',      s=0, label='$\sigma_\phi$'),
                                plt.scatter([], [], marker='o',color='gold',       s=s_square[0], label='<0.1'),
                                plt.scatter([], [], marker='o', color='violet',    s=s_square[1], label='>0.1'),
                                plt.scatter([], [], marker='o', color='limegreen', s=s_square[2], label='>0.15'),
                                plt.scatter([], [], marker='o', color='dodgerblue',s=s_square[3], label='>0.2')]
                
            plt.imshow(thresh, 'gray')
            plt.title(f'Date = 08.02.2019     Time: {hhmm}')
            # plt.legend(handles=legend_elements, loc='best')
            plt.show()
            
            
            plt.savefig(pic_path+'08022019__'+hhmm+'.png')
            plt.show()  
            
            
            # i = 0
            # for i in hhmm: 
            #     plt.savefig(f"{pic_path}01012019__{hhmm}_{i:4}.png")
            #     i += 1
            
            
           
                      
           
txt_path.close()






       # # show our output images
       # cv2.imshow("Sobel/Scharr X", gX)
       # cv2.imshow("Sobel/Scharr Y", gY)
       # cv2.imshow("Sobel/Scharr Combined", combined)
       # cv2.waitKey(0) 
    
    
    
   
    
    
    
    
    
        ### plots both the input image and the threshold image 
        # plt.subplot(211),plt.imshow(img)
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(212),plt.imshow(thresh, 'gray')
        # plt.imsave(r'thresh.png',thresh)
        # plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])
        # plt.tight_layout()
        # plt.show()
    
    
    
    
    
  
    
