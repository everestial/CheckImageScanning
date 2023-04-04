import cv2

def increase_contrast_and_grayscale(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast by applying a simple histogram equalization
    contrast_image = cv2.equalizeHist(gray_image)
    
    return contrast_image


def increase_contrast(image):
    # Load the input image
    image = cv2.imread(image)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase the contrast of the image
    contrast_image = cv2.equalizeHist(gray_image)

    # Apply a Gaussian blur to the image
    blur_image = cv2.GaussianBlur(contrast_image, (5, 5), 0)

    # Convert the image to black and white
    black_and_white_image = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return black_and_white_image


def increase_contrast_and_grayscale_clahe(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE to the grayscale image
    contrast_image = clahe.apply(gray_image)
    
    return contrast_image

# Example usage:
# contrast_gray_image = increase_contrast_and_grayscale("input_image.jpg")
# contrast_gray_image = increase_contrast_and_grayscale("0004_line_3.png")

# contrast_gray_image = increase_contrast_and_grayscale_clahe("0004_line_3.png")
contrast_gray_image = increase_contrast("0004_line_3.png")



cv2.imwrite("0004_line_3_contrast.png", contrast_gray_image)
