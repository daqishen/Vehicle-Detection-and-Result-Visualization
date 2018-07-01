#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""



# change direction

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import xlrd
import _init_paths
import os, cv2
os.chdir("D:/NYU/Image_and_Video_Processing/project/tf-faster-rcnn-master/tools")
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
import qytools as qy
import folium
import pandas as pd
from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import base64
import argparse
from vgg16 import vgg16
#from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from bbox_transform import clip_boxes, bbox_transform_inv


#CLASSES = ('__background__',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('__background__',
           'car', 'bus', 'van', 'others',
           )
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def num_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    num_vehicles = np.sum(len(inds))
    print(num_vehicles)
    return num_vehicles

def demo(sess, net, image_name,direction='Temp'):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, direction, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s '.format(timer.total_time))


    # Visualize detections for each class
    CONF_THRESH = 0.3
    NMS_THRESH = 0.6 #numaximum surpression
    
    
    a = [0,0,0,0]
    i = 0
    
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
#        num_vehicle = num_vehicle + num_detections(im, cls, dets, thresh=CONF_THRESH)
        a[i] = a[i]+num_detections(im, cls, dets, thresh=CONF_THRESH)
        

    return a
def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])
#    tfmodel = 'output\\res101\\voc_2007_trainval+voc_2012_trainval\\default\\res101_faster_rcnn_iter_110000.ckpt'   

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    
    

    
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
        
   
#    net.create_architecture("TEST", 35,
#                          tag='default', anchor_scales=[8, 16, 32])
    a = net.create_architecture("TEST", 5,
                          tag='default', anchor_scales=[8, 16, 32])

#%%
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    
    
#%%
    maps = xlrd.open_workbook('../../map.xlsx')  #get the camera information
    table = maps.sheets()[0]
    camera =  np.array(table.col_values(0)[1:])
    urls = np.array(table.col_values(2)[1:])
    feature_num = np.array(table.col_values(3)[1:])
    title = table.row_values(0)
    latitude = np.array(table.col_values(4)[1:])
    longitude = np.array(table.col_values(5)[1:])
    camera_total = np.vstack((camera,urls,feature_num,latitude,longitude))
#%%
    print('Loaded network {:s}'.format(tfmodel))
    #calculate the cameras to be processed
    j = 0
    for i in range(camera_total[1].shape[0]):
        if camera_total[2][j] == '':
            camera_total = np.delete(camera_total,j,1)
            
        else:
            j=j+1
        
        

#%%            
    num_pro = 5 #process image number in each folder
    name = 9
    bias = 5*name
    cars = np.zeros(camera_total[1].shape[0])
    im_names = np.zeros(num_pro+bias).astype(str)
    for i in range(num_pro):
        im_names[i] = str(i+bias).zfill(6)+'.jpg'
        

#%% calculate vehicles number in each camera. Skip the broken images(regard as the previous image)
    for i in range(10):
        a = 0
        for j in range(num_pro):
            direction = 'location/'+camera_total[0][i]+'/Download'
            if os.path.isfile(os.path.join(cfg.DATA_DIR, direction, im_names[j])):
                a = np.sum(demo(sess,net,im_names[j],direction))
                cars[i] = cars[i]+a
            else:
                cars[i] = cars[i]+a

#%%
    camera_total = np.vstack((camera_total,cars))

#%%
    np.save(str(name)+".npy",camera_total)
#%% draw map
#    schoolMap_test = folium.Map(location=[40.78315, -73.9712], zoom_start=12)
#    df_camera = pd.DataFrame(camera_total)
#    S = (32,3)
#    df = pd.DataFrame(np.zeros(S))
#    busy_lvl = np.zeros(df_camera.shape[1])
#    #calculate busy level for each camera
#    for i in range(df_camera.shape[1]):
#        if float(df_camera[i][6]) > 0:
#            busy_lvl[i] = float(df_camera[i][5])/(num_pro * float(df_camera[i][6]))
#    gp=df_camera.T.groupby(by=2)
#    newdf=gp.size()  
#    index = newdf.index.tolist()
#    for i in range(len(index)):
#        df[2][int(float(index[i]))] = newdf[index[i]]
#        
#    for i in range(df_camera.shape[1]):
#        dist = int(float(df_camera[i][2]))
#        df[1][dist] = df[1][dist]+float(busy_lvl[i])
#    for i in range(df.shape[0]):
#        df[0][i] = i+1
#        
#    alp = 150
#    df = df.rename(columns={0: 'district'})
#    df = df.rename(columns={1: 'Mean Scale Score'})
#    for i in range(df.shape[0]):
#        if df[2][i] >0:
#            df['Mean Scale Score'][i] = df['Mean Scale Score'][i]/df[2][i]*alp
#
#    num_base = 20 #base number in threshold
#    threshold_scale = [0,num_base,2*num_base,3*num_base,4*num_base,5*num_base]
#    
#    schoolMap_test.choropleth(geo_data="map_qy.json",
#                         fill_color='YlOrRd', fill_opacity=0.5, line_opacity=0.5,
#                         threshold_scale = threshold_scale,
#                         data = df,
#                         key_on='feature.id',
#                         columns = ['district', 'Mean Scale Score']                          
#                         )
#    
#    for i in range(camera_total[0].shape[0]):
#        lon = float(camera_total[4][i])
#        lat = float(camera_total[3][i])
#        tit = camera_total[0][i]
#        folium.Marker([lat,lon],
#              popup=tit,
#              icon=folium.Icon(color='green')
#              ).add_to(schoolMap_test)
#    schoolMap_test.save(outfile='cars_heatmap_test.html')

    

    