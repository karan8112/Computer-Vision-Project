# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 18:36:01 2018

@author: karan
"""
import os
from StaticSalient import static_saliency
from DynamicSalient import dynamic_saliency
from VIdeoFrames import extractImages


def process(load_data_set):
    video_image_data_path = [dataset for dataset in os.listdir(load_data_set) if
                        os.path.isdir(load_data_set)]
    static_saliency(video_image_data_path, load_data_set)
    dynamic_saliency(video_image_data_path, load_data_set)

def main():
    global load_data_set
    print("Enter 1 for testing your own video \nEnter 2 for testing the dataset \nEnter h for help")
    k = input("Please enter your choice:")
    if(k == '1'):
        video_path = input("Input the path of the video:")
        video_output = './video/%s'%os.path.basename(video_path[0:-4]) + '/'
        if not  os.path.exists(video_output):
            os.makedirs(video_output)
        extractImages(video_path, video_output)
        load_data_set = './video'
        process(load_data_set)
    elif(k == '2'):
        load_data_set = input("Input path of image dataset:")
        process(load_data_set)
    elif(k == 'h'):
        print("To run the program you can use the given dataset.\nFirst of all you have to select the choice on which you have to apply salient object detection i.e. on your own video or on datasets. \nNow you have to provide the path of the dataset. \nThen the proram take all the names of video in single array and program starting to apply saliet object deteection. \n After completion of salient object detection he automatically starting to apply dynamic saliency on video frames. \nAt the completion of program you will see two folder static_saliency_output and dynamic_saliency_output where you will be able to see the output of your dataset of every video.")
        
    

if __name__ == '__main__':
    main()