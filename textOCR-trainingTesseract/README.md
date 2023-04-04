## Tesseract Training Tutorial    

<br>

### Some source tutorials   
Repository mentioned in https://youtu.be/KE4xEzFGSU8 (but only slightly helpful)


## Dependencies

Install tesseract with the following dependencies.
```
$ tesseract --version
tesseract 4.1.1
 leptonica-1.82.0
  libgif 5.1.9 : libjpeg 8d (libjpeg-turbo 2.1.1) : libpng 1.6.37 : libtiff 4.3.0 : zlib 1.2.11 : libwebp 1.2.2 : libopenjp2 2.4.0
 Found AVX2
 Found AVX
 Found FMA
 Found SSE
 Found libarchive 3.6.0 zlib/1.2.11 liblzma/5.2.5 bz2lib/1.0.8 liblz4/1.9.3 libzstd/1.4.8
```

## Installation

Install Tesseract OCR on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install -y libtesseract-dev
```

For other systems, please refer to Tesseract's official installation guide:
https://tesseract-ocr.github.io/tessdoc/Installation.html

<br>

Install Leptonica (required by Tesseract for training):
```bash
sudo apt-get install -y libleptonica-dev
```
<br>

Create Python (version >= 3.8x) virtual environment:
```
python -m venv .envForTesseractTraining

# activate the environment
source .envForTesseractTraining/bin/activate
```
NOTE: Alternatively we can setup this training environment`pipenv` or `conda`.

<br>

Install required Python libraries:
```bash
pip install pytesseract
pip install opencv-python
pip install opencv-python-headless
pip install Wand
pip install pillow
pip install imutils
```

Install ImageMagick (optional):
```bash
sudo apt-get install imagemagick
```
**Please note** that the commands listed above are mainly for Ubuntu/Debian systems. For other operating systems, you might need to adjust the installation commands accordingly.

After installing Tesseract and its dependencies, you can use the training guideline (below) for creating the training data (preprocess images, and extract text from images) and preparing the custom Tesseract model.

<br>

## Training

Training involves several steps.

### Step 1: Prepare images for training    
**1a) Convert image to greyscale**  
  Since text detection involves only identifying character by the lines, color is less important (but we could add images with color later, it if helps with model tuning).

**1b) Split the image into text lines (or image lines)**    
  The goal is to make training images small and controllable. Tesseract-5 also strictly requires line images, and it is helpful to use line images for Tesseract-4 (current version) as well. The same line images can be used for training with Tesseract-5.

**NOTE:**
- Either of the step 1a, 1b or both the steps, of preparing the training can done manually **(actually this is preferable at the beginning)**.    
- For automated training data preparation use the script `prepare_training_images.py`.
- Generally it is better to have training images in 300 DPI, but it is not strict requirement. However, it is said that 300 DPI images make training better. **We can test this assumptions during our training**.
<br>

**MANUAL method:**    
For image preparation and edit use:
  - https://convert.town/image-dpi
  - https://tech-lagoon.com/imagechef/en/image-to-monochrome.html
  - local image manipulation applciation like: paint, windows image viewer, GIMP etc.

<br>

**AUTOMATED method:**    
Use the script `prepare_training_data.py`.
```bash
$ python prepare_training_images.py input_image_path output_folder split_input_image
# $ python prepare_training_images.py ./original_images/0001b.jpg ./line_images False
```
**NOTE:**    
Read the training images prepared using step 1a, 1b (depending upon the image) or just give the path to the original image or skip this step if manual image prep was good enough.

<br>

### Step 2    
**2a) Create box files and ground truth**    
  Tesseract uses the given training images (actually line images) along with the box file, ground truth text and LSTM file to further train the pre-trained model.

Create box files for training using line images from folder line_images.
```bash
$ python prepare_training_data.py generate_box_files input_folder output_folder
# $ python prepare_training_data.py generate_box_files ./line_images ./training_box_gttext_data

OUTPUT:
Tesseract Open Source OCR Engine v4.1.1 with Leptonica
Tesseract Open Source OCR Engine v4.1.1 with Leptonica
Tesseract Open Source OCR Engine v4.1.1 with Leptonica
Empty page!!
Empty page!!
Tesseract Open Source OCR Engine v4.1.1 with Leptonica
```
NOTE: 
  - "Empty page!!" means that the image was not processed properly by Tesseract, because
  it was too blurry, low contrast etc. 
  - These images can be either 
    - removed. Also make sure to remove *.gt.txt and *.png and *.box files for these images that were not processed by Tesseract. 
  - or pre-processed to prepare for retraining.
  - or we can create *.gt.txt and *.box files for these images manually. 
    Just create a ground truth by reading the text in the image. Then create an empty *.box file. Then open the *.png file by jtessboxedEditor (mentioned below) which will allow you to add the box co-oridnates and text.   
<br>

**2b) Correct the box and ground truth file (fully and strictly a manual task)**    
Now, correct the data (box and ground truth) using some available tools.
  - https://zdenop.github.io/qt-box-editor/
  - https://github.com/zdenop/qt-box-editor
  - https://github.com/nguyenq/jtessboxeditor/releases
  - https://groups.google.com/g/tesseract-ocr/c/T5m7ICIcYk4
  - https://groups.google.com/g/tesseract-ocr/c/A1Qq_vfKyRs/m/Y2qDh1jtjQAJ
    
NOTE:   
  - I use jtessboxedEditor
  - keep `images` and `box`, `*gt.txt` data together to edit the box files.
  - for the images that were not processed by tesseract (on **Step 2a**), we may remove those from the training or further modify the details in the box file or ground truth text file.
  
<br>

### Step 3: Generate `*.lstm` files

**NOTE:** LSTM file should be prepared only after the correction of box and gt.txt files. Make sure the training data is very accurate and proof read several times. Any low quality data within the training data should be removed.
<br>

**CMD line for creating LSTM for a single file, directly using tesseract:**
```bash
$ tesseract line_0_b.png line_0_b -c tessedit_create_boxfile=1 lstm.train
# CONSOLE OUTPUT -> 
# Tesseract Open Source OCR Engine v4.1.1 with Leptonica
```

**Or, create LSTM for all the prepared files using the script `prepare_training_data.py`.**
```bash
$ python prepare_training_data.py generate_lstm_files input_folder output_folder
# $ python prepare_training_data.py generate_lstm_files ./line_images ./line_images
```

```bash
$ python prepare_training_data.py generate_lstm_files ./trainingData2 ./trainingData2
# CONSOLE OUTPUT ->
# Generated ./trainingData2/0001b_line_1.lstmf
# Tesseract Open Source OCR Engine v4.1.1 with Leptonica
# Generated ./trainingData2/0001_line_13.lstmf
# Tesseract Open Source OCR Engine v4.1.1 with Leptonica
# ...
# ...
```

### Step 4: Training begins

**04-a) Create a list file**   
After creating a list of `*.lstmf` files to be used for training, create a file called `list.txt`, that contains the paths of all the `*.lstmf` files.
This can be done manually, or use one of the following commands to generate the list automatically:

```bash
# 1)
find . -name '*.lstmf' -exec echo {} \; > list.txt
# find . -name 'training_box_gttext_data/*.lstmf' -exec echo {} \; > list.txt

# 2)
ls *.lstmf > list.txt
# ls training_box_gttext_data/*.lstmf > list.txt
```

**04-b) Download a pre-trained model (pre-DONE):**    
Download a pre-trained model, such as eng.traineddata, 
from the Tesseract GitHub repository: https://github.com/tesseract-ocr/tessdata_best

Or, the a trained file could be inside the installed tesseract directory too if a full install was done.

NOTE: here we are using `eng.traineddata` as a base model (trained model) to further train the model, which is already available in the folder `my_trained_model`. If new file is need from a different file do as needed.

**04-c) Extract `.lstm` file from pre-trained model (pre-DONE):**    
Extract the `.lstm` file from the pre-trained model using the following command:
```bash
$ combine_tessdata -e eng.traineddata eng.lstm
Extracting tessdata components from eng.traineddata
Wrote eng.lstm
Version string:4.00.00alpha:eng:synth20170629:[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]
17:lstm:size=11689099, offset=192
18:lstm-punc-dawg:size=4322, offset=11689291
19:lstm-word-dawg:size=3694794, offset=11693613
20:lstm-number-dawg:size=4738, offset=15388407
21:lstm-unicharset:size=6360, offset=15393145
22:lstm-recoder:size=1012, offset=15399505
23:version:size=80, offset=15400517
```
NOTE: A file named `eng.lstm` should be created in the current directory or given output path. A LSTM file `eng.lstm` is already extracted and put in the folder `my_trained_model`. If new file is need from a different file do as needed.

**04-d) Now train the model**    
```bash
$ lstmtraining \
    --model_output my_trained_model/mytrainedmodel \
    --continue_from eng.lstm \
    --traineddata eng.traineddata \
    --train_listfile list.txt \
    --max_iterations 40
Loaded file my_trained_model/mytrainedmodel_checkpoint, unpacking...
Successfully restored trainer from my_trained_model/mytrainedmodel_checkpoint
Loaded 1/1 lines (1-1) of document trainingData2/0001_line_13.lstmf
Loaded 1/1 lines (1-1) of document trainingData2/0002_line_2.lstmf
Loaded 1/1 lines (1-1) of document trainingData2/0001b_line_1.lstmf
Loaded 1/1 lines (1-1) of document trainingData2/0004_line_2.lstmf
Loaded 1/1 lines (1-1) of document trainingData2/0004_line_5.lstmf
Loaded 1/1 lines (1-1) of document trainingData2/0004_line_6.lstmf
Loaded 1/1 lines (1-1) of document trainingData2/0004_line_4.lstmf
At iteration 29/40/40, Mean rms=1.688%, delta=4.832%, char train=10.881%, word train=70%, skip ratio=0%,  wrote checkpoint.

Finished! Error rate = 10.881
```
NOTE:    
The above process should created `checkpoint` files after each run. We will need to use the latest (check DATE-TIME STAMP) of the file that has lowest error rate.

<br>

**04-e) Stop the training**

```bash
lstmtraining \
    --stop_training \
    --continue_from my_trained_model_checkpoint \
    --traineddata /usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata \
    --model_output my_trained_model.traineddata

# lstmtraining \
#     --stop_training \
#     --continue_from my_trained_model/mytrainedmodel10.881_29.checkpoint \
#     --traineddata eng.traineddata \
#     --model_output my_trained_model/my_trained_model.traineddata
```
NOTE: There must be a trained model file named `my_trained_model.traineddata` in the model output path.

<br>

### Step 05: Test/use the trained model for OCR.    
Now let's test/use the trained model for OCR.    

**05-a) Copy or set the path to the trained model**
```bash
# 1) you can either copy the model to the model directory path for the installed tesseract 
# copy to:  Ubuntu-18.04\usr\share\tesseract-ocr\4.00\tessdata\trained_model.traineddata

$ cp my_trained_model/trained_model1.traineddata /usr/share/tesseract-ocr/4.00/tessdata/
# NOTE: use `sudo` if permission denied

# 2) or set the path
$ export TESSDATA_PREFIX=./my_trained_model/my_trained_model.traineddata 
```

**05-b) Test/Use the model**

```bash
## writes output the file: text_with_basemodel.txt
$ tesseract line_3_b.png text_with_basemodel -l eng
Tesseract Open Source OCR Engine v4.1.1 with Leptonica
output: Mia X. Hailew

## to write the output to console, use stdout
$ tesseract line_3_b.png stdout -l eng
Mia X. Hailey

## lets now try custom trained model
$ tesseract line_3_b.png text_with_cmodel -l my_trained_model
Tesseract Open Source OCR Engine v4.1.1 with Leptonica
Output: MiaX. Hailey


## let's try on few more files that I used for training
$ tesseract 0004_line_2.png stdout -l eng
Hr: Sweetheart .--:-

$ tesseract 0004_line_2.png stdout -l trained_model1
Hii Sweetheart....-

## NOTE: the result is much better with the trained_model. Well, but it's the same data used for training.

## But, let's use a different data not used in training but which has similar font.
$ tesseract 0004_line_3.png stdout -l eng
Yow are very Special for me -

$ tesseract 0004_line_3.png stdout -l trained_model1
You are very seecialforrme

## NOTE: we see improvement in OCR characters, but some got worse. 
## All, we need to do it improve box file annotation and training.
```

**Step 05-a (Automated Testing) - using automated models comparison**

To compare the performance of the default model and the recently prepared model, you can use both models to perform OCR on a set of test images and compare their output against the ground truth. You can use a Python script to automate the process and compute the performance metrics like character error rate (CER) and word error rate (WER).

Here's a simple example of how to do this:

  - Prepare a set of test images with corresponding ground truth text files (e.g., image_1.png and image_1_gt.txt).

  - Run OCR using the default model and the custom model, and save the outputs to separate text files.

  - Calculate the CER and WER for both models by comparing their OCR outputs to the ground truth.

**Use the Python script `compare_models.py`**
```bash
$ python compare_models.py 
Tesseract Open Source OCR Engine v4.1.1 with Leptonica
Tesseract Open Source OCR Engine v4.1.1 with Leptonica
Default Model - Average WER: 0.16666666666666666 Average CER: 0.16666666666666666
Custom Model - Average WER: 1.0 Average CER: 1.0

# NOTE: 
# - low WER, CER value is good, 0 being the best.
# - since I only trained on small amount of trained data the trained data was not good, I am getting high WER and CER.
```
