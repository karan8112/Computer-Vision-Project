# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 18:46:41 2018

@author: karan
"""

from keras.layers import Input
from keras.models import Model
import os
from keras.optimizers import SGD
import numpy as np
from model import static_vgg
from scipy.misc import imread, imsave, imresize

img_channel_mean = [103.939, 116.779, 123.68]

def static_saliency(video_data_object,directory_path):
    #load model
    print("dsdd")
    x = Input(batch_shape=(1, 224, 224, 3))
    print("batch:",x)
    #model = VGG16(include_top=True, weights = 'imagenet')
    #model.save('h1.h5')
    static_saliency_model = Model(inputs=x, outputs = static_vgg(x))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    static_saliency_model.compile(loss="binary_crossentropy",optimizer=sgd,metrics=["accuracy"])
    print("fdsd")
    print(static_saliency_model.summary())
    #val = static_saliency_model.evaluate(x, static_vgg(x), verbose = 1, steps = None)
    #print(val)
    #static_saliency_model.save_weights("static_saliency.h5")
    static_saliency_model.load_weights("static_vgg.h5")
    
    for video_data in video_data_object:
        video_output_path = './static_saliency_output/' + video_data
        #check the folder pr directory exist else create new one
        if not  os.path.exists(video_output_path):
            os.makedirs(video_output_path)
        print("path created")
        print(video_data)
        #prepare test data for testing
        images = [directory_path + '/' + video_data + '/' + i for i in os.listdir(directory_path + '/' + video_data) if
                        i.endswith(('.jpg', '.jpeg', '.png'))]
        images.sort()
        print("Images path:", images)
        test_data_image = []
        for j in images:
            annotation_data = {'image': j}
            test_data_image.append(annotation_data)
        print("created array of image frame",test_data_image)
        #calculate static saliency of testing images
        for data in test_data_image:
            temp_im = np.zeros((1,224,224,3))
            print("temp image:",temp_im)
            img = imread(data['image'])
            img_name = os.path.basename(data['image'])
            print("image name:",img_name)
            r_img = imresize(img, (224, 224))
            r_img = r_img.astype(np.float32)
            r_img[:, :, 0] -= img_channel_mean[0]
            r_img[:, :, 1] -= img_channel_mean[1]
            r_img[:, :, 2] -= img_channel_mean[2]
            r_img = r_img[:, :, ::-1]
            temp_im[0, :] = np.copy(r_img)
            predictions = static_saliency_model.predict(temp_im, batch_size=1)
            static_saliency = postprocess_predictions(predictions[2][0, :, :, 0], img.shape[0],
                                                               img.shape[1])
            imsave(video_output_path + '/%s.png' % img_name[0:-4], static_saliency.astype(int))


def postprocess_predictions(pred, shape_r, shape_c):
    print("proessing")
    pred = imresize(pred, (shape_r, shape_c))
    print(np.max(pred))
    pred = pred / np.max(pred) * 255.
    return pred
            
    