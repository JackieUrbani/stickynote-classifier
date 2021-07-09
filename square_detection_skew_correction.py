import cv2
import numpy as np
import os

images_directory = r'images/'
list_of_files = []

for filename in os.listdir(images_directory):
    list_of_files.append(filename)

image_number = 0
for file in list_of_files:
    image = cv2.imread(images_directory + file)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen,160,255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    close = cv2.bitwise_not(close)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 6000
    max_area = 60000
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            ROI = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            
            # Correct orientation
            # Locate corners
            i = 0
            max = 0
            max_i = 0
            for top_pixel in ROI[0]:
                if top_pixel > max:
                    max = top_pixel
                    max_i = i
                i = i + 1
            top_max = max_i

            i = 0
            max = 0
            max_i = 0
            for bottom_pixel in ROI[len(ROI) - 1]:
                if bottom_pixel > max:
                    max = bottom_pixel
                    max_i = i
                i = i + 1
            bottom_max = max_i

            max = 0
            max_j = 0
            for j in range(len(ROI)):
                pixel = ROI[j, 0]
                if pixel > max:
                    max = pixel
                    max_j = j
            left_max = max_j

            max = 0
            max_j = 0
            for j in range(len(ROI)):
                pixel = ROI[j, len(ROI[0]) - 1]
                if pixel > max:
                    max = pixel
                    max_j = j
            right_max = max_j

            # Locate points of the documents or object which you want to transform
            pts1 = np.float32([[top_max,0], [len(ROI[0]),right_max], [0,left_max], [bottom_max,len(ROI)]])
            pts2 = np.float32([[0, 0], [128, 0], [0, 128], [128, 128]])
              
            # Apply Perspective Transform Algorithm
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(ROI, matrix, (128, 128))
            #cv2.imshow('Corrected', result)
            
            cv2.imwrite('processed-images/corrected_{}.png'.format(image_number), result)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            image_number += 1

    #cv2.imshow('sharpen', sharpen)
    #cv2.imshow('close', close)
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('image', image)
    #cv2.waitKey()