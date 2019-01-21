# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 01:08:40 2018

@author: karan
"""

from __future__ import division
from keras.layers import Input
from keras.models import Model
import os
import numpy as np
from keras.optimizers import SGD
from model import dynamic_vgg
from scipy.misc import imread, imsave, imresize




img_channel_mean = [103.939, 116.779, 123.68]

def dynamic_saliency(video_data_object,directory_path):
    a1 = Input(batch_shape=(1, 224, 224, 3))
    a2 = Input(batch_shape=(1, 224, 224, 3))
    a3 = Input(batch_shape=(1, 224, 224, 1))
    dynamic_saliency_model = Model(inputs=[a1, a2, a3], outputs=dynamic_vgg([a1, a2, a3]))
    dynamic_saliency_model.load_weights('dynamic_vgg.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    dynamic_saliency_model.compile(loss="binary_crossentropy",optimizer=sgd,metrics=["accuracy"])
    for video_data in video_data_object:
        video_output_path = './dynamic_saliency_output/' + video_data
        #check the folder pr directory exist else create new one
        if not  os.path.exists(video_output_path):
            os.makedirs(video_output_path)
        print("path created")
        print(video_data)
        #take first image i.e. 0 index image
        im1 = [directory_path + '/' + video_data + '/' + i for i in os.listdir(directory_path + '/' + video_data)[0:-1] if
                        i.endswith(('.jpg', '.jpeg', '.png'))]
        #take next image i.e. 0+1 index image
        im2 = [directory_path + '/' + video_data + '/' + i for i in os.listdir(directory_path + '/' + video_data)[1:] if
                        i.endswith(('.jpg', '.jpeg', '.png'))]
        #staic saliency mage is sam as first image
        static_saliency_im = ['./static_saliency_output/' + video_data + '/' + i for i in os.listdir('./static_saliency_output/' + video_data)[0:-1] if
                        i.endswith(('.jpg', '.jpeg', '.png'))]
        
        im1.sort()
        im2.sort()
        static_saliency_im.sort()
        
        #create array to store test data image
        test_data_image = []
        for i, j, k in zip(im1, im2, static_saliency_im):
            annotation_data = {'image1': i, 'image2': j, 'static_saliency': k}
            test_data_image.append(annotation_data)
        print("created array of image frame")
        
        #now calculate dynamic static saliency
        for data in test_data_image:
            temp_im1 = np.zeros((1, 224, 224, 3))
            temp_im2 = np.zeros((1, 224, 224, 3))
            static_temp = np.zeros((1, 224, 224, 1))
            img1 = imread(data['image1'])
            img2 = imread(data['image2'])
            static_saliency = imread(data['static_saliency'])
            img_name = os.path.basename(data['image1'])
            r_img1 = imresize(img1, (224, 224))
            r_img1 = r_img1.astype(np.float32)
            r_img1[:, :, 0] -= img_channel_mean[0]
            r_img1[:, :, 1] -= img_channel_mean[1]
            r_img1[:, :, 2] -= img_channel_mean[2]
            r_img1 = r_img1[:, :, ::-1]
            temp_im1[0, :] = np.copy(r_img1)
            r_img2 = imresize(img2, (224, 224))
            r_img2 = r_img2.astype(np.float32)
            r_img2[:, :, 0] -= img_channel_mean[0]
            r_img2[:, :, 1] -= img_channel_mean[1]
            r_img2[:, :, 2] -= img_channel_mean[2]
            r_img2 = r_img2[:, :, ::-1]
            temp_im2[0, :] = np.copy(r_img2)
            static_image = imresize(static_saliency, (224, 224))
            static_image = static_image.astype(np.float32)
            static_temp[0, :, :, 0] = np.copy(static_image)
            res = [temp_im1, temp_im2, static_temp]
            predictions = dynamic_saliency_model.predict(res, batch_size=1)
            static_saliency = postprocess_predictions(predictions[2][0, :, :, 0], img1.shape[0],
                                                               img1.shape[1])
            imsave(video_output_path + '/%s.png' % img_name[0:-4], static_saliency.astype(int))
            
            
def postprocess_predictions(pred, shape_r, shape_c):
    print("proessing")
    pred = imresize(pred, (shape_r, shape_c))
    print(np.max(pred))
    pred = pred / np.max(pred) * 255
    return pred
            