# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:38:35 2018

Create file

"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
def createfile(path,file_name,exist):
    """
    The path is and file name should be string
    exist depend on whether to cover the existed file or not
    exist should be True or Falses
    """
    wholepath = path+'/'+file_name
    pathlib.Path(wholepath).mkdir(parents=True, exist_ok=exist) 
def count_vehicles(file):
    '''
    count vehecles in txt file
    the output is the vehicles number in each images
    '''
    df = pd.read_csv(file, header=None)
    df[0].value_counts(sort=False)
    vehicles = df[0].value_counts(sort=False) # index starts from 1
    return vehicles

def print_vehicles_in_time(vehicles,fps):
    '''
    print vehicles distribution in time 
    '''
    bias = 0
    num = int(vehicles.shape[0]/fps)
    x = np.linspace(1,num,num)
    y = np.zeros(num)
    for i in range(num):
        for j in range(fps):
            if vehicles[fps*i+j+1+bias] >=0:                
                y[i] = y[i]+vehicles[fps*i+j+1+bias]
            else:
                bias+=1
    plt.stem(x,y)
    plt.title('Vehicle Distribution')
    plt.xlabel('time')
    plt.ylabel('Vehicle number')
    plt.show()
    plt.close()
    return y 
    
def plot_distribution(txt,freq,savepath,bias,time_s,time_e):
    '''
    txt is the file name
    freq = fps
    time_s = start time
    time_e = end time
    '''
    df = pd.read_csv(txt, header=None)
    img_order = np.array(df[0])
    vehicles_num = np.array(df[10])
    vehicles_per_img = np.zeros(max(img_order))
    num = int(max(img_order)/freq)
    x = np.linspace(time_s,time_e,num)
    y = np.zeros(num)
    
    for i in range(vehicles_num.shape[0]):
        vehicles_per_img[img_order[i]-1] = vehicles_per_img[img_order[i]-1] + vehicles_num[i]
    non_zero = np.nonzero(vehicles_per_img)
    if vehicles_per_img[0] ==0:
        vehicles_per_img[0] = non_zero[0][0]
    for j in range(vehicles_per_img.shape[0]):
        if vehicles_per_img[j] == 0:
            vehicles_per_img[j] = vehicles_per_img[j-1]

    for i in range(num):
        for j in range(freq):
            y[i] = y[i]+vehicles_per_img[freq*i+j]
    plt.plot(x,y)
    plt.title('Vehicle Distribution')
    plt.xlabel('time')
    plt.ylabel('Vehicle number')
    matplotlib.pyplot.savefig(savepath)
    plt.close()
    return y
def plot_histogram(series,savepath,bins):
    '''
    txt is the file name
    freq = fps
    time_s = start time
    time_e = end time
    '''
    his = pd.Series(series)
    his.plot.hist(stacked=True, bins=bins)
    plt.title('histogram')
    matplotlib.pyplot.savefig(savepath)
    plt.close()

   
# TEST 
#path = '../project'
#file_name = 'dictionary67'
#exist = False # whether to replace the existed file
#createfile(path,file_name,exist)
    
    

    