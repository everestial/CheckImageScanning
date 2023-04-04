

## base code for creating box file
# tesseract output_image.tif test -c tessedit_create_boxfile=1

import os
import subprocess
import shutil

def generate_box_files(input_folder, output_folder):
    input_files = os.listdir(input_folder)

    for file in input_files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            input_file_path = os.path.join(input_folder, file)
            file_name, _ = os.path.splitext(file)
            output_base = os.path.join(output_folder, file_name)

            # first let's copy the original line images inside this folder
            # this is done to keep the image, box, gt.txt, .lstm file all at one place to do the optimization
            shutil.copyfile(input_file_path, os.path.join(output_folder, file))

            # Run Tesseract to generate box files
            box_command = [
                'tesseract',
                input_file_path,
                output_base,
                '-c', 'tessedit_create_boxfile=1'
            ]
            subprocess.run(box_command)

            # Run Tesseract to generate the text file
            # NOTE: only if generating ground truth.text as a separate text file
            text_command = [
                'tesseract',
                input_file_path,
                output_base
            ]
            subprocess.run(text_command)

            # Rename the recognized text file to the ground truth file
            recognized_text_file = f'{output_base}.txt'
            ground_truth_file = f'{output_base}.gt.txt'
            os.rename(recognized_text_file, ground_truth_file)


def generate_lstm_files(input_folder, output_folder):
    input_files = os.listdir(input_folder)

    for file in input_files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            input_file_path = os.path.join(input_folder, file)
            file_name, _ = os.path.splitext(file)
            output_base = os.path.join(output_folder, file_name)


            # Generate LSTM file
            lstm_command = [
                "tesseract", input_file_path, output_base, "lstm.train"]
            print(f"Generated {output_base}.lstmf")
            subprocess.run(lstm_command)

import sys

task = sys.argv[1]
input_folder = sys.argv[2]
output_folder = sys.argv[3]

## Step 1: 
## let's create box files for training using line images in folder line_images
if task == 'generate_box_files':
    # generate_box_files('line_images', 'training_box_gttext_data')
    generate_box_files(input_folder, output_folder)


## Step 2 (Manual Task):
## then correct the data (box and ground truth) using some available tools
    # https://zdenop.github.io/qt-box-editor/
    # https://github.com/zdenop/qt-box-editor
    # https://github.com/nguyenq/jtessboxeditor/releases
    # NOTE: 
        # I use jtessboxedEditor
        # keep images and box, gt.txt data together.


## Step 3: generate *.lstm files
# NOTE: lstm should be prepared only after the correction of box and gt.txt files.
# Here, therefore I have put a breakpoint (which should be managed someother way later)
# CMD line for generating *.lstm for a single file:
# $ tesseract line_0_b.png line_0_b -c tessedit_create_boxfile=1 lstm.train
# Tesseract Open Source OCR Engine v4.1.1 with Leptonica
elif task == 'generate_lstm_files':
    # generate_lstm_files('training_box_gttext_data', 'training_box_gttext_data')
    generate_lstm_files(input_folder, output_folder)


