
import cv2

# FUNCTION for: reshaping the image to 320w x 320h
    # Function to: Resize the input image to the desired dimensions without preserving its aspect ratio.
    # this is done because EAST model only takes image in this resolution
    # we can move the work of this function to another module (process_image.py) in the future
def reformat_image_resolution(image_path, new_width=None, new_height=None):
    """
    Resize the input image to the desired dimensions without preserving its aspect ratio.

    :param image: str, path to the input image
    :param width: int, desired width of the resized image
    :param height: int, desired height of the resized image
    :return: tuple, resized image and its dimensions
    """
    input_image_obj = cv2.imread(image_path)
    original_image_obj = input_image_obj.copy()
    original_height, original_width = input_image_obj.shape[:2]

    # find the ratio of original:requested resolution
        # ACRONYM: ratio_width_O2N -> ratio of width Original to New
    ratio_width_O2N = original_width / float(new_width)
    ratio_height_O2N = original_height / float(new_height)

    output_image_obj = cv2.resize(input_image_obj, (new_width, new_height))
    new_height, new_width = output_image_obj.shape[:2]

    # cv2.imwrite("001_i_resized.jpg", image)
    # # cv2.imwrite("images/lebron_james_resized.jpg", image)
    # # TODO: revisit this problem: the image is blurred and is squeezed, fix this?

    return (output_image_obj, original_image_obj, 
            original_width, original_height, 
            ratio_width_O2N, ratio_height_O2N, 
            new_width, new_height)


###########################################################################
################    TEST (Remove the docstring and call)    ###############
"""
inFile = '001_i.jpg'

## Step 02: Reshape the image before passing it to EAST model
    # resize the image before passing it to EAST
output_image_obj, original_image_obj, original_width, original_height, \
    ratio_width_O2N, ratio_height_O2N, new_width, new_height = \
        reformat_image_resolution(image_path = inFile, new_width=320, new_height=320)
print("Image Values:\n original_height, original_width, ratio_width_O2N, ratio_height_O2N, new_height, new_width \n", 
    original_height, original_width, ratio_width_O2N, ratio_height_O2N, new_height, new_width)

# write the reshaped file
cv2.imwrite("001_O_reshaped.jpg", output_image_obj)
"""

## Call with virtual environment activated
# $ python reformat_image_resolution.py 

## Expected output:
  # Image Values:
  # original_height, original_width, ratio_width_O2N, ratio_height_O2N, new_height, new_width 
  # 744 1212 3.7875 2.325 320 320
