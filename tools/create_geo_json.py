# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:24:24 2018
create geo_json file


@author: qiyue
"""

import xlrd
import os
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt



os.chdir("D:/NYU/Image_and_Video_Processing/project")
maps = xlrd.open_workbook('map.xlsx')  #get the camera information
os.chdir("D:/NYU/Image_and_Video_Processing/project/test")
table = maps.sheets()[0]
camera =  np.array(table.col_values(0)[1:])
urls = np.array(table.col_values(2)[1:])
feature_num = np.array(table.col_values(3)[1:])
latitude = np.array(table.col_values(4)[1:])
longitude = np.array(table.col_values(5)[1:])
camera_total = np.vstack((camera,urls,feature_num,latitude,longitude))

sites = np.array([1.3,2.4])
for camera_num in range (camera.shape[0]):
    if camera_total[4][camera_num] != '':
        long = camera_total[4][camera_num]
        lat  = camera_total[3][camera_num]
        sites = np.vstack((sites,np.array([float(long),
                                          float(lat)])))
sites = np.delete(sites, 0,0)

tri = Delaunay(sites)
plt.triplot(sites[:,0], sites[:,1], tri.simplices.copy())
plt.plot(sites[:,0], sites[:,1], 'o')
plt.show()
sites[tri.simplices]