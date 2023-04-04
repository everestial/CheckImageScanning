
import json
import numpy as np

import cv2
import numpy as np

from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
# import skimage.io as io
import matplotlib.pyplot as plt
import os
import pytesseract
import sys
from argparse import ArgumentParser

import argparse
import os
import shutil
from pprint import pprint

def read_image(input):
    if isinstance(input, str):
        img = cv2.imread(input)
        if img is None:
            print(f"Error: Failed to read the image from {input}")
        else:
            print(f"Image read from file path: {input}")
    elif isinstance(input, np.ndarray):
        img = input
        print("Image read from an existing OpenCV object (numpy array).")
    else:
        print("Error: Unsupported input type.")
        img = None

    return img


# FUNCTION for: detecting text instances in the input image
## NOTE: this function can be updated to train the EAST as per our image requirements
def detect_text_instances_using_EAST(image, east, min_confidence, width = 320, height = 320):

    # image =  cv2.imread(image)
    image = read_image(image)
    if not image.shape[0:2] == (320, 320):
        print("The image file is not of the shape (320, 320). \
              Please provide image with appropriate dimension.")
        print("Exiting image scanning.")
        sys.exit(0)

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first one is to generate the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east)

    # construct a blob from the image and then perform a forward pass of the model to obtain the two output layer sets
    # blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
    #     (123.68, 116.78, 103.94), swapRB=True, crop=False)

    blob = cv2.dnn.blobFromImage(image, 1.0, (width, height),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
        
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rectangles, confidences) = decode_predictions(scores, geometry, min_confidence)
    boxes = non_max_suppression(np.array(rectangles), probs=confidences)

    return boxes



## FUNCTION for: decode the predictions on the identified boxes and using confidence for filtering
def decode_predictions(scores, geometry, min_confidence):
    (numRows, numCols) = scores.shape[2:4]    
    rectangles = []
    confidences = []
    
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rectangles.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rectangles, confidences)


# FUNCTION is: to find the co-ordinates of the detection but on the original image
def find_bounding_box_coordinates(
        original_width, original_height, 
        ratio_width_O2R, ratio_height_O2R,
        original_image, padding,
        boxes = None, scan_text = False):
    
    # initialize the list of boxes_and_text
    boxes_and_text = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective ratios
        startX = int(startX * ratio_width_O2R)
        startY = int(startY * ratio_height_O2R)
        endX = int(endX * ratio_width_O2R)
        endY = int(endY * ratio_height_O2R)

        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)

        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        # endX = min(origW, endX + (dX * 2))
        # endY = min(origH, endY + (dY * 2))
        endX = min(original_width, endX + (dX * 2))
        endY = min(original_height, endY + (dY * 2))

        ## TODO: move this to a separate function (and optimize text extraction there)
        # NOTE: we can directly extract the text in this ROI (which are smaller)
        # OR, we can merge the close bounding boxes and extract text later

        # extract the actual padded ROI (region of interest)
        roi = original_image[startY:endY, startX:endX]
        #roi = cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)  # if interested in GrayScaling

        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        # config = ("-l eng --oem 3 --psm 11")
        config = r'-c tessedit_char_whitelist=0123456789 --psm 6'

        text = pytesseract.image_to_string(roi,config=config)
        
        # add the bounding box coordinates and OCR'd text to the list
        # of boxes_and_text
        boxes_and_text.append(((startX, startY, endX, endY), text))

    # sort the boxes_and_text bounding box coordinates from top to bottom
    # here it is sorted by the second value of the firt element of the list
    """
    e.g: its is sorted by 51, 58, 77, 84
    coordinate parameters are: 
    startX, startY, endX, endY | x1, y1, x2, y2
    [
    ((1053, 51, 1157, 87), 'oool\n\x0c'),
    ((65, 58, 152, 92), 'Mia X. F\n\x0c'),
    ((544, 77, 659, 113), 'Rime\n\x0c'),
    ((148, 84, 306, 120), 'JR ADDRESS\n\x0c'),
    ...
    ]
    """
    boxes_and_text = sorted(boxes_and_text, key=lambda r:r[0][0])

    # # converting the tuples into list for later manipulation
    for i,x in enumerate(boxes_and_text):
        boxes_and_text[i] = list(boxes_and_text[i])
        boxes_and_text[i][0] = list(boxes_and_text[i][0])

    # extracting a separate box (hoping it will be useful?)
    boxes_only = [x[0] for x in boxes_and_text]

    return boxes_and_text, boxes_only


def text_OCR_using_pytesseract():
    pass


##################################################################################
################    TEST (Remove the docstring quotes and call)    ###############
"""
## STEP 02: FIND TEXT INSTANCES USING EAST (EASY AND ACCURATE SCENE TEXT DETECTOR)
## NOTE: 
# this step uses the DNN (Deep Neural Network) pretrained model
# this step/model can be 
    # updated or retrained as needed, or
    # replaced with other models in the future

## Step 02-A: pass the reshaped image to EAST model and get text boxes
# only the reshaped image (320 x 320) is valid for EAST model
# NOTE: there is another model of EAST that takes in 1200 x 1200 image (but that is for later)
# that could give us some good boxes_and_text


## Read required images
original_image_obj = read_image("001_i.jpg")  
# or, some saved binary object after reading from pickle file
# original_image_obj = pickled_obj(read_image("001_i.jpg"))

reshaped_image_obj = "001_i_reshaped.jpg"  
# or, some saved binary object after reading from pickle file
# reshaped_image_obj = pickled_obj(read_image("001_i_reshaped.jpg"))

## Other required parameters
east = "frozen_east_text_detection.pb"
min_confidence = 0.5
#nms_thresh = 0.24
width = 320
height = 320
padding = 0.09

## get values based on original and reshaped image
## NOTE: hard coding here that is based on the original and reshaped image of "001_i.jpg"
(original_height, original_width, ratio_width_O2N, 
 ratio_height_O2N, new_height, new_width) = (744, 1212, 3.7875, 2.325, 320, 320)

boxes = detect_text_instances_using_EAST(image=reshaped_image_obj, east=east, min_confidence=min_confidence, width=320, height=320)

print("boxes")
pprint(boxes)
print()
    
## Step 02-B: 
    # transform the bounding boxes detected by EAST (on reshaped image) onto original image 
    # at the same time search for text if needed.
boxes_and_text, boxes_only = \
    find_bounding_box_coordinates(
        original_width, original_height, 
        ratio_width_O2N, ratio_height_O2N,
        original_image_obj, padding,
        boxes = boxes, scan_text = False)

print("boxes_and_text")
pprint(boxes_and_text)
# breakpoint()

#################################################################
## function for writing data to json. 
# NOTE: it's not used in main application but only used for debugging.
def ndarray_to_json_file(array, file_name):
    with open(file_name, 'w') as json_file:
        if type(array) is np.ndarray:
            json.dump(array.tolist(), json_file, indent=4)
        else: # if the array is actually a string or list (not a true array)
            json.dump(array, json_file, indent=4)
    print(f"Successfully saved ndarray to {file_name}")

def json_file_to_ndarray(file_name):
    with open(file_name, 'r') as json_file:
        array = np.array(json.load(json_file))
    print(f"Successfully loaded ndarray from {file_name}")
    return array

# Example usage
# array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# json_file_name = "array.json"

# Convert ndarray to JSON file
# ndarray_to_json_file(array, json_file_name)
ndarray_to_json_file(boxes, "001_reshaped_image_instance_coordinates.json")
ndarray_to_json_file(boxes_and_text, "001_original_image_instance_coordinates_and_text.json")


# Read JSON file to ndarray
# loaded_array = json_file_to_ndarray(json_file_name)
# print("Loaded array:\n", loaded_array)
loaded_boxes = json_file_to_ndarray("001_reshaped_image_instance_coordinates.json")
loaded_boxes_and_text = json_file_to_ndarray("001_reshaped_image_instance_coordinates.json")


## Call with virtual environment activated
# $ python find_text_instances.py 
## Expected output:
    # Image read from file path: 001_i.jpg
    # Image read from file path: 001_i_reshaped.jpg
    # [INFO] loading EAST text detector...
    # [INFO] text detection took 0.341679 seconds
    # [[[10, 629, 293, 682], 'ROUTING\n\x0c'], [[33, 674, 263, 731], 'number\n\x0c'], 
    # [[53, 374, 154, 408], 'MEMO\n\x0c'], [[64, 290, 169, 329], 'five hu\n\x0c'],
    # ...
    # [[1053, 51, 1157, 87], 'oool\n\x0c'], [[1084, 153, 1170, 183], 'DATE\n\x0c']]
    # Successfully saved ndarray to 001_reshaped_image_instance_coordinates.json
    # Successfully saved ndarray to 001_original_image_instance_coordinates_and_text.json
    # Successfully loaded ndarray from 001_reshaped_image_instance_coordinates.json
    # Successfully loaded ndarray from 001_reshaped_image_instance_coordinates.json
"""