import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Histrogram equilization to enhance the images
class Histrogram_equilizer:
    def __init__(self):
        pass
    
    def equilize(self, image_path, output_folder):
        image = cv2.imread(image_path)

        #coversion of rgb image to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Histogram equilization
        equilizer = cv2.equalizeHist(gray_img)

        # Get the file extension of the input image
        _, extension = os.path.splitext(image_path)

        #to show the difference
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_equilized' + extension
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, equilizer)


#function calling - Histogram equilization to enhance the denoised image quality to see the regions better
if __name__ == '__main__':
    input_folder = "D:\\master_thesis\\datasets\\denoised\\normal"
    output_folder = "D:\\master_thesis\\datasets\\histogram_equilization\\normal"

#histogram equilization 
equilization = Histrogram_equilizer()

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        equilization.equilize(image_path, output_folder)






