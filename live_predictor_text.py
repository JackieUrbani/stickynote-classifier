from keras.models import model_from_json
import cv2
import numpy as np
import os
import time

# Load json and create model
json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# Load weights into new model
model.load_weights("model2.h5")
print("Loaded model from disk")
# Define labels
labels = ['throwaway','hi','there,',"I'm",'Jackie',':-)']

def predict(image):
    #testing_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testing_img = np.array(image) / 255
    testing_img = testing_img.reshape(1,128,128,1)
    prediction = model.predict(testing_img)
    #print(prediction)
    i = 0
    max_val_i = 0
    max_val = prediction[0,0]
    for label in prediction[0]:
        if label > max_val:
            max_val_i = i
            max_val = label
        i = i + 1
    return labels[max_val_i]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    
    count = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            ROI = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            
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
            if (predict(result) == labels[1]):
                cv2.imshow('1', result)
            elif (predict(result) == labels[2]):
                cv2.imshow('2', result)
            elif (predict(result) == labels[3]):
                cv2.imshow('3', result)
            elif (predict(result) == labels[4]):
                cv2.imshow('4', result)
            elif (predict(result) == labels[5]):
                cv2.imshow('5', result)
            
            # Write out prediction for square found
            if (predict(result) != "throwaway"):
                print(str(count) + ": " + predict(result))
                
            count = count + 1
    #time.sleep(0.5)
    
cv2.destroyAllWindows()