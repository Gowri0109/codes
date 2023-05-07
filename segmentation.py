import cv2
import os
import pandas as pd
import numpy as np

#Segmentation using region of interest (ROI)
class segment_roi:
    def __init__(self):
        self.image = None
        self.roi_pts = []
        self.completed = False

#to select the region of interest 
    def select_roi(self, image):
        self.image = image
        clone = self.image.copy()
        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select Roi", self._select_roi_callback)

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
        return image
    
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
roi_selector = segment_roi()

#input and output folder path
input_folder = "D:\master_thesis\datasets\histogram_equilization\glioma"
output_folder = "D:\master_thesis\datasets\segmented\glioma"
filename = ()

# Create output directory if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#loop over all the images in the input folder
if filename in os.listdir(input_folder):
    #get filename from the filepath
    file_path = os.path.join(input_folder, filename)
    #load images
    image_path = cv2.imread(file_path)

    #to select the roi
    roi_pts = segment_roi.select_roi(image_path)

#to extract roi
    if roi_pts is not None:
        mask = np.zeros(image_path.shape[:2], dtype=np.unit8)
        roi_corners = np.array(roi_pts,dtype=np.int32)
        cv2.fillPoly(mask, [roi_corners], {255,255,255})
        roi = cv2.bitwise_and(image_path, image_path, mask=mask)
        #save in different folder
        for filename in os.listdir(input_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                output_path = os.path.join(input_folder, filename)
                cv2.imwrite(os.path.join(output_folder), roi_pts)
                print("Saved output image to:", output_path)
                cv2.imshow("ROI", roi)
                cv2.waitKey(0)
