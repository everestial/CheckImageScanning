import cv2, pytesseract

## FUNCTION for: writing the bounding boxes and text on the image
def write_bounding_boxes(image, bounding_boxes_and_texts, write_text = False, rescan_text = False):

    # making a copy so there is no write over on the object being passed among functions
    output_image = image.copy()
    # if writing on a gray scale
    # output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2GRAY)  

    # loop over the boxes_and_text
    for ((startX, startY, endX, endY), text) in bounding_boxes_and_texts:
        # write the bounding box based on the co-ordinates
        cv2.rectangle(output_image, (startX, startY), (endX, endY),(0, 255, 0), 2)

        if rescan_text:
            # NOTE: we can directly extract the text in this ROI (which are smaller)
            # OR, we can merge the close bounding boxes and extract text later

            # extract the actual padded ROI (region of interest)
            # roi = original_image[startY:endY, startX:endX]
            roi = output_image[startY:endY, startX:endX]
            #roi = cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)  # if interested in GrayScaling

            # in order to apply Tesseract v4 to OCR text we must supply
            # (1) a language, (2) an OEM flag of 4, indicating that the we
            # wish to use the LSTM neural net model for OCR, and finally
            # (3) an OEM value, in this case, 7 which implies that we are
            # treating the ROI as a single line of text
            # config = ("-l eng --oem 3 --psm 11")
            config = r'-c tessedit_char_whitelist=0123456789 --psm 6'
            # text = pytesseract.image_to_string(roi,config=config)
            text = pytesseract.image_to_string(roi).strip()

            # NOTE: this text can also be stored to be write later to a text file or excel sheet

            # add the bounding box coordinates and OCR'd text to the list
            # of boxes_and_text
            # boxes_and_text.append(((startX, startY, endX, endY), text))

        # write text on top of the bounding box
        # the writing position is shifted by 5 (denoted by starY - 5)
        if write_text:
            # strip out non-ASCII text so we can draw the text on the image
            # using OpenCV, then draw the text and a bounding box surrounding
            # the text region of the input image
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.putText(output_image, text, (startX, startY - 5),cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)

    return output_image


## TODO: Complete this test
###########################################################################
################    TEST (Remove the docstring and call)    ###############
"""
## Step 02-C (Optional): Write the bounding boxes and text on the images
    # This is optional step and done as necessary, bc we may be just interested in the final bounding box and texts
    # But, these steps are kepts here, and can be used if necessary
    ## write bounding box to a file (on original image dimension)

# TODO: fix these variable values too
original_image_obj = None
boxes_and_text = None
file_name = None
file_extension = None

original_image_obj_with_bounding_boxes = write_bounding_boxes(
    image = original_image_obj, bounding_boxes_and_texts=boxes_and_text, write_text = False, rescan_text=False)
# output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)  # if writing on a gray scale
# cv2.imwrite("001_O_east03_with_bounding_boxes.jpg", original_image_obj_with_bounding_boxes)

# NOTE: if writing text on the boxes
original_image_obj_with_bounding_boxes_and_text = write_bounding_boxes(
    image = original_image_obj, bounding_boxes_and_texts=boxes_and_text, write_text = True, rescan_text=True)
# cv2.imwrite("001_O_east03_with_bounding_boxes_and_text.jpg", original_image_obj_with_bounding_boxes_and_text)

"""


