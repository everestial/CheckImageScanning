

# NOTE: to continue work use this - NOTE NOTE NOTE

# NOTE: based on https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/ 
# Install required packages
  # pip install --upgrade imutils

    # main tutorials
  # https://jaafarbenabderrazak-info.medium.com/opencv-east-model-and-tesseract-for-detection-and-recognition-of-text-in-natural-scene-1fa48335c4d1

    # other tutorials
    # https://www.folio3.ai/blog/text-detection-by-using-opencv-and-east/
    # https://medium.com/technovators/scene-text-detection-in-python-with-east-and-craft-cbe03dda35d5

    # image rotation
    # https://github.com/argman/EAST/issues/368

  # contains code for rotating image
    # https://github.com/cvkworld/OCR

    # NOTE: also check the folder name OCR-master

# import necessary packages
# from email.mime import image
# from turtle import width
# from unicodedata import east_asian_width
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
import pandas as pd

import argparse
import os
import shutil
from pprint import pprint

from reformat_image_resolution import reformat_image_resolution
from find_text_instances import detect_text_instances_using_EAST, find_bounding_box_coordinates
from write_bounding_box import write_bounding_boxes
from merge_bounding_boxes import merge_close_bounding_boxes

def main():
    parser = argparse.ArgumentParser(description="Process input and output files.")
    
    parser.add_argument("inFile", help="Input file path", type=str)
    parser.add_argument("-o", "--outFile", help="Output file path (optional)", type=str, default=None)
    
    args = parser.parse_args()
    
    # process_files(args.inFile, args.outFile)
    inFile, outFile = args.inFile, args.outFile

    # out_filename, infile_extension = process_files(args.inFile, args.outFile)
    in_filename, infile_extension, out_filename = process_files(inFile, outFile)
    ocrd_text = [inFile] 
    # 'infile_name\traw_texts\ttexts_on_bb\ttexts_on_merged_bb'

    ## Step?: extract very first raw text without any manipulation of image
    ocrd_text.append("")  # just adding empty string for now # TODO: fix later
    
    #### Step 01: set variables and read image file
    # set variables and argument 
    # input_image = '001_i.jpg'  # REMOVE
    east = 'frozen_east_text_detection.pb'
    # original_image = '001_i.jpg'


    # set threshold value
    # NOTE: we are using east model that takes in 320 x 320 input.
    # there is another model that takes in 1200 x 1200 which we can use if need be in the future.
    min_confidence = 0.5
    #nms_thresh = 0.24
    width = 320
    height = 320
    padding = 0.09


    ## Step 01-C: Reshape the image before passing it to EAST model
    # resize the image before passing it to EAST
    reshaped_image_obj, original_image_obj, original_width, original_height, \
        ratio_width_O2N, ratio_height_O2N, new_width, new_height = \
            reformat_image_resolution(image_path = inFile, new_width=320, new_height=320)
    print("Image Values:\n original_height, original_width, ratio_width_O2N, ratio_height_O2N, new_height, new_width \n", 
        original_height, original_width, ratio_width_O2N, ratio_height_O2N, new_height, new_width)

    # write the reshaped file
    # cv2.imwrite("001_O_reshaped.jpg", output_image_obj)
    cv2.imwrite(out_filename + '_reshaped' + '.png', reshaped_image_obj)


    #################################################################################
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

    # The `reshaped_image_obj` is passed to the EAST model to detect text instances.
    # Other required parameter values are also passed.
    boxes = detect_text_instances_using_EAST(image=reshaped_image_obj, east=east, min_confidence=min_confidence, width=320, height=320)
    # print("boxes")
    # print(boxes)
        
    ## Step 02-B: 
        # transform the bounding boxes detected by EAST (on reshaped image) onto original image 
        # at the same time search for text if needed.
    boxes_and_text, boxes_only = \
        find_bounding_box_coordinates(
            original_width, original_height, 
            ratio_width_O2N, ratio_height_O2N,
            original_image_obj, padding,
            boxes = boxes, scan_text = False)
    
    # Step ?: extracting a separate text only (hoping it will be useful?)
    texts_only_from_hc_bb = [x[1] for x in boxes_and_text]
    ocrd_text.append(texts_only_from_hc_bb)
    
    # print(boxes_and_text)
    # breakpoint()

    ## Step 02-C (Optional): Write the bounding boxes and text on the images
    # This is optional step and done as necessary, bc we may be just interested in the final bounding box and texts
    # But, these steps are kepts here, and can be used if necessary
    ## write bounding box to a file (on original image dimension)
    original_image_obj_with_bounding_boxes = write_bounding_boxes(
        image = original_image_obj, bounding_boxes_and_texts=boxes_and_text, write_text = False, rescan_text=False)
    # output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)  # if writing on a gray scale
    # cv2.imwrite("001_O_east03_with_bounding_boxes.jpg", original_image_obj_with_bounding_boxes)
    cv2.imwrite(out_filename + '_with_bounding_boxes' + infile_extension, original_image_obj_with_bounding_boxes)


    # NOTE: if writing text on the boxes
    original_image_obj_with_bounding_boxes_and_text = write_bounding_boxes(
        image = original_image_obj, bounding_boxes_and_texts=boxes_and_text, write_text = True, rescan_text=True)
    # cv2.imwrite("001_O_east03_with_bounding_boxes_and_text.jpg", original_image_obj_with_bounding_boxes_and_text)
    cv2.imwrite(out_filename + '_with_bounding_boxes_and_text' + infile_extension, original_image_obj_with_bounding_boxes_and_text)

    ## Step 03: Now merge the bounding boxes that are close to each other

    ## Step 03-A: Merge the boxes
    ## NOTE: The prarmeters for merge (e.g. h_distance, v_distance) can be defined in another nested function within the function called.
    # merged_boxes_and_text, merged_boxes_only = merge_close_bounding_boxes(boxes_and_text, h_distance = 0, v_distance = 0)
    merged_boxes_and_text, merged_boxes_only = merge_close_bounding_boxes(boxes_and_text)
    
    # extracting a separate text only (hoping it will be useful?)
    texts_only_from_hc_merged_bb = [x[1] for x in merged_boxes_and_text]
    ocrd_text.append(texts_only_from_hc_merged_bb)
    # breakpoint()

    # Step?: write all the extracted data to a file
    # Convert the list of lists to a DataFrame
    df = pd.DataFrame([ocrd_text], columns=['out_filename', 'raw_texts', 'texts_on_bb', 'texts_on_merged_bb'])
    df.to_csv(out_filename + "_OCRD_data" + ".txt", sep='\t', header=True, index=False)

    # print(merged_boxes_and_text)
    # print()
    # print(texts_only_from_hc_merged_bb)
    # breakpoint()

    ## Step 03-B (Optional): Again, write the merged boxes and the texts to the image 
    # write the output image
    original_image_obj_with_merged_bounding_boxes = write_bounding_boxes(original_image_obj, merged_boxes_and_text, write_text = False, rescan_text=False)
    # output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)  # if writing on a gray scale
    # cv2.imwrite("001_O_east03_merged_bounding_boxes.jpg", original_image_obj_with_merged_bounding_boxes)
    cv2.imwrite(out_filename + '_with_merged_bounding_boxes' + infile_extension, original_image_obj_with_merged_bounding_boxes)

    original_image_obj_with_merged_bounding_boxes_and_text = write_bounding_boxes(original_image_obj, merged_boxes_and_text, write_text = True, rescan_text=True)
    # cv2.imwrite("001_O_east03_merged_bounding_boxes_with_text.jpg", original_image_obj_with_merged_bounding_boxes_and_text)
    cv2.imwrite(out_filename + '_with_merged_bounding_boxes_and_text' + infile_extension, original_image_obj_with_merged_bounding_boxes_and_text)

    print(f"\nCompleted OCR of the image {inFile}")
    print(f"Output data written in file with prefix {out_filename}")
    # NOTE: CONTINUE ... from here ....
    ## TODO:
    # Text cleaning with python 
    # Text labeling using NLP

# import cv2
# import numpy as np

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



def process_files(inFile, outFile):
    if not os.path.isfile(inFile):
        print(f"Error: {inFile} does not exist.")
        return

    in_filename, infile_extension = os.path.splitext(inFile)

    if outFile is None:
        # in_filename, infile_extension = os.path.splitext(inFile)
        # outFile = f"{file_name}_output{infile_extension}"
        out_filename = in_filename
        return in_filename, infile_extension, out_filename
    else: 
        out_filename, _ = os.path.splitext(outFile)
        return in_filename, infile_extension, out_filename

    # shutil.copy(inFile, outFile)
    # print(f"Successfully copied {inFile} to {outFile}")




if __name__ == "__main__":
    main()
    sys.exit()

#### UPTO HERE SO FAR 


# TODO: categorize the extracted text
# use custom of NLP library