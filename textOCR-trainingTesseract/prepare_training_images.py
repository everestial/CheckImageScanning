import cv2
import numpy as np
from wand.image import Image
import sys, os

def prepare_training_images(image_path, output_dir, split = False, deskew=False):
    # get directory, file name and file extension
    directory, file_name, file_extension = extract_path_info(image_path)

    # Load image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # TODO: Also add method to deskew image


    if not split or split == 'False':          
        ## update image pixel and save the file
        #1. first convert OpenCV image object to image.wand object
        wand_image_obj = opencv_image_to_wand_image(image)

        # breakpoint()
        #2. now update the DPI (dots per inch) and save the file
            # NOTE: if needed, increase pixel density any image directly updated using `convert`
            # convert split_images/line_0.png -units PixelsPerInch -density 300 split_images/line_0_dpi300.png
        wand_img_300_dpi = change_image_dpi(wand_image_obj, target_dpi=300)
        wand_img_300_dpi.save(filename= f"{output_dir}/{file_name}.png")
        print("Successfully updated image DPI")
      
    else:
        # Pre-process the image
        # Apply bilateral filter for noise reduction
        image = cv2.bilateralFilter(image, 9, 75, 75)

        # Resize image to increase resolution
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        ## if image contrast needed first
        # image = enhance_contrast(image)

        # Binarize the image using Otsu's method
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        lines = compute_horizontal_projection(binary)

        ## if no lines were extracted from image
        if len(lines) == 0:
            print("No text lines were extracted from the given input image.")

            ## try reducing white noise and increasing contrast to find text/outline 
                
            # remove white noise from image
            # noise_removed_image = remove_noise(image)
            image = remove_noise(image)

            # set a contrasted image to handle blurred images
            # bw_image = enhance_contrast(image)
            image = enhance_contrast(image)
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            ## get lines after image
            lines = compute_horizontal_projection(binary)

        # cv2.imwrite(f"{output_dir}/{file_name}.png", image)
        # exit()

        if len(lines) == 0:
            print("No text lines extracted from the given image.")
            return

        # breakpoint()
        # Save each line as a separate image
        for i, (start, end) in enumerate(lines):
            line_image = image[start:end, :]
            # cv2.imwrite(f"{output_dir}/line_{i}.png", line_image)

            ## update image pixel before writing
            #1. first convert OpenCV image object to image.wand object
            wand_image_obj = opencv_image_to_wand_image(line_image)

            # breakpoint()
            #2. now update the DPI (dots per inch) and save the file
                # NOTE: if needed, increase pixel density any image directly updated using `convert`
                # convert split_images/line_0.png -units PixelsPerInch -density 300 split_images/line_0_dpi300.png
            wand_img_300_dpi = change_image_dpi(wand_image_obj, target_dpi=300)
            # wand_img_300_dpi.save(filename= f"{output_dir}/line_{i}.png")
            wand_img_300_dpi.save(filename= f"{output_dir}/{file_name}_line_{i}.png")

        # print("Successfully splitted image into image lines")
    

def compute_horizontal_projection(binary):
    # Compute horizontal projection profile
    projection = np.sum(binary, axis=1)

    # Find the upper and lower boundaries of text lines
    lines = []
    in_line = False
    for i, val in enumerate(projection):
        if not in_line and val > 0:
            in_line = True
            start = i
        elif in_line and val == 0:
            in_line = False
            end = i
            lines.append((start, end))
    return lines


## remove white noise
def remove_noise(image):
    # Convert the image to grayscale if it is not already
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Apply binary thresholding to convert the image to black and white
    _, bw_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Perform morphological opening to remove small white noise
    noise_removed_image = cv2.morphologyEx(bw_image, cv2.MORPH_OPEN, kernel)

    return noise_removed_image


## increase image contrast
def enhance_contrast(image):
    # Convert the image to grayscale if it is not already
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply adaptive thresholding to enhance the contrast and convert to black and white
    bw_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return bw_image


def opencv_image_to_wand_image(opencv_img):
    # Convert the OpenCV image (BGR) to RGB
    img_rgb = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    
    # Create a Wand image object
    wand_img = Image.from_array(img_rgb)
    return wand_img


def change_image_dpi(input_image, target_dpi=300):
    img = input_image
    img.units = 'pixelsperinch'
    img.resolution = (target_dpi, target_dpi)
    return img 


def extract_path_info(file_path):
    """
    # Example usage
    # file_path = '/path/to/your/file.txt'
    # directory, file_name, file_extension = extract_path_info(file_path)

    # print("Directory:", directory)
    # print("File name:", file_name)
    # print("File extension:", file_extension)
    """
    directory, file = os.path.split(file_path)
    file_name, file_extension = os.path.splitext(file)
    
    return directory, file_name, file_extension


# input_image_path = 'split_images/line_0.png'
# output_image_path = 'split_images/line_0_dpi300.png'
# change_image_dpi(input_image_path, output_image_path)

# Example usage
# split_lines("input_image.png", "output_directory")
# split_lines("output_image.tif", "split_images")

# split_lines("original_images/0001.jpg", "line_images")
# print("Successfully splitted image into image lines")

input_image = sys.argv[1]
output_folder = sys.argv[2]
split = sys.argv[3]

prepare_training_images(input_image, output_folder, split)

# $ python split_imagelines.py ./original_images/0002.jpg line_images False
# Successfully splitted image into image lines







