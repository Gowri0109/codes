import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Histrogram equilization to enhance the images
class Histrogram_equilizer:
    def __init__(self, imagepath):
        self.imagepath = imagepath
    
    def equilize(self, image_path, output_folder):
        image = cv2.imread(image_path)

        #coversion of rgb image to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Histogram equilization
        equilizer = cv2.equalizeHist(gray_img)

        #to show the difference
        ouput_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_folder, equilizer)


#function calling - Histogram equilization to enhance the denoised image quality to see the regions better
if __name__ == '__main__':
    input_folder = "D:\master_thesis\datasets\denoised\glioma"
    output_folder = "D:\master_thesis\datasets\histogram_equilization\glioma"

#histogram equilization 
equilization = Histrogram_equilizer()

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        equilization.equilize(image_path, output_folder)



#Segmentation using thresholding or ROI
class segment:
    def __init__(self):
        self.image = None
        self.roi_pts = []
        self.completed = False

#to select the region of interest 
    def select_roi(self, image):
        self.image = image
        clone = self.image.copy()
        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select Roi",self.select_roi_callback)

        while True:
            cv2.imshow("Select ROI", self._draw_roi(clone))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                self.roi_pts = []
                clone = self.image.copy()
            elif key == ord("c"):
                if len(self.roi_pts) == 4:
                    self.completed = True
                    cv2.destroyWindow("Select ROI")
                    return self.roi_pts
                else:
                    print ("Select four points to define ROI:")
            elif key ==27:
                cv2.destroyWindow("Select ROI")
                break
        return None
    
    def _select_roi_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.roi_pts) < 4:
                self.roi_pts.append((x,y))

#to draw the region of interest points required 
    def _draw_roi(self, image):
        for pt in self.roi_pts:
            cv2.circle(image, pt, 5, (0, 255, 0),2)

        if len(self.roi_pts) == 4:
            cv2.line(image, self.roi_pts[0], self.roi_pts[1], (0,255,0),2)
            cv2.line(image, self.roi_pts[1], self.roi_pts[2], (0,255,0),2)  
            cv2.line(image, self.roi_pts[2], self.roi_pts[3], (0,255,0),2)  
            cv2.line(image, self.roi_pts[3], self.roi_pts[0], (0,255,0),2)  
        return image         

#funtion calling - to select the region of interest in the input data

#create roi_selector object
roi_selector = segment()

#input and output folder path
input_folder = "D:\master_thesis\datasets\histogram_equilization\glioma"
output_folder = "D:\master_thesis\datasets\segmented\glioma"
i= ()

#loop over all the images in the input folder
if i in os.listdir(input_folder):
    #load images
    image_path = os.path.join(input_folder, i)
    image = cv2.imread(image_path)

#to select the roi
roi_pts = segment.select_roi(input_folder)

#to extract roi
if roi_pts is not None:
    mask = np.zeros(image.shape[:2], dtype=np.unit8)
    roi_corners = np.array(roi_pts,dtype=np.int32)
    cv2.fillPoly(mask, [roi_corners], {255,255,255})
    roi = cv2.bitwise_and(image, image, mask=mask)
    #save in different folder
    cv2.imwrite(os.path.join(output_folder), roi_pts)
    #cv2.imshow("ROI", roi)
    cv2.waitKey(0)



