import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Histrogram equilization to enhance the images
class Histrogram_equilizer:
    def __init__(self, imagepath):
        self.imagepath = imagepath
    
    def equilize(self);
        image = cv2.imread(self.imagepath)

        #coversion of rgb image to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Histogram equilization
        equilizer = cv2.equalizeHist(gray)

        #to show the difference
        cv2.imshow('Original', image)
        cv2.imshow('Equilized', equilizer)
        cv2.waitKey(0)
        cv2.destroyAllWindows

    
       