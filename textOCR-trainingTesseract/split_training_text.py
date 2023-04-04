import os
import random
import pathlib
import subprocess

training_text_file = 'langdata/eng.training_text'

lines = []

with open(training_text_file, 'r') as input_file:
    for line in input_file.readlines():
        lines.append(line.strip())

# output_directory = 'tesstrain/data/Apex-ground-truth'
# output_directory = 'tesstrain/data/DejavuSans-ground-truth'
output_directory = 'tesstrain/data/Arial-ground-truth'



if not os.path.exists(output_directory):
    os.mkdir(output_directory)

random.shuffle(lines)

count = 100

lines = lines[:count]

line_count = 0
for line in lines:
    training_text_file_name = pathlib.Path(training_text_file).stem
    line_training_text = os.path.join(output_directory, f'{training_text_file_name}_{line_count}.gt.txt')
    with open(line_training_text, 'w') as output_file:
        output_file.writelines([line])

    file_base_name = f'eng_{line_count}'

    # subprocess.run([
    #     'text2image',
    #     # '--font=Apex',
    #     '--font=Arial',
    #     f'--text={line_training_text}',
    #     f'--outputbase={output_directory}/{file_base_name}',
    #     '--max_pages=1',
    #     '--strip_unrenderable_words',
    #     '--leading=32',
    #     '--xsize=3600',
    #     '--ysize=480',
    #     '--char_spacing=1.0',
    #     '--exposure=0',
    #     '--unicharset_file=langdata/eng.unicharset'
    # ])

    subprocess.run([
    'text2image',
    # '--font=Apex',
    '--font=Arial',
    f'--text={line_training_text}',
    f'--outputbase={output_directory}/{file_base_name}',
    '--max_pages=1',
    '--strip_unrenderable_words',
    '--leading=32',
    '--xsize=3600',
    '--ysize=280',
    '--char_spacing=1.0',
    '--exposure=0',
    '--unicharset_file=langdata/eng.unicharset'
    ])

    line_count += 1

## some working commands

## Convert any fonts file to image (tiff) and generate boxfile
# text2image --text=test_text.txt --outputbase=output_image --fonts_dir=/usr/share/fonts --font='Arial' --fontconfig_tmpdir=/tmp

# text2image --text=test_text.txt --outputbase=output_image --fonts_dir=/usr/share/fonts --font='Arial' --fontconfig_tmpdir=/tmp --max_pages=1' --strip_unrenderable_words' --leading=32' --xsize=3600' --ysize=480'

## using local fonts folder
# $ text2image --text=test_text.txt --outputbase=output_image --fonts_dir=./fonts_collection --font=Arial --fontconfig_tmpdir=/tmp --max_pages=1 --strip_unrenderable_words --leading=32 --xsize=3600 --ysize=480
# Rendered page 0 to file output_image.tif


## create training file
# $ tesseract output_image.tif test box.train
# Tesseract Open Source OCR Engine v4.1.1 with Leptonica
# Page 1
# APPLY_BOXES:
#    Boxes read from boxfile:      29
#    Found 29 good blobs.
# Generated training data for 7 words

## create box file
# tesseract output_image.tif test -c tessedit_create_boxfile=1


## split the image into image lines
# python split_imagelines.py 
# TODO:setup arguments for input/output folder

## now, lets create data from split images
# $ tesseract split_images/line_0.png test -c tessedit_create_boxfile=1
# Tesseract Open Source OCR Engine v4.1.1 with Leptonica
# Warning: Invalid resolution 0 dpi. Using 70 instead.
# Estimating resolution as 467
# NOTE: there was issue with the resolution 
# regardless of the resolution issue there will be data (box file and others).
"""
Would it matter if I left the image as is, at the existing resolution?
- Leaving the image at its existing resolution will not cause any major issues, 
but it might affect the OCR accuracy to some extent. Tesseract uses the DPI information 
to estimate the image resolution and scale the image appropriately during OCR processing. 
When the DPI value is incorrect or missing, Tesseract's ability to recognize the text 
might be reduced, especially if the characters are very small or have a low resolution.

Using the default DPI value of 70 might be sufficient for some images, 
but it may not provide the best results in all cases. If you are satisfied with the 
OCR results you are getting with the current DPI setting, you can continue using the 
image as-is. However, if you notice that the OCR accuracy is not as high as you would like, 
it would be a good idea to set the correct DPI value for the image, as this may improve the results.
"""

## So, lets fix that image resolution issue.
# $ convert split_images/line_0.png -units PixelsPerInch -density 300 split_images/line_0_dpi300.png

## create box file again
# $ tesseract split_images/line_0_dpi300.png line_0_test -c tessedit_create_boxfile=1
# Tesseract Open Source OCR Engine v4.1.1 with Leptonica


