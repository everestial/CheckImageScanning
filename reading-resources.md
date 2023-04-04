# Project Check Image Scanning and Data Extraction
Application to scan the check image and extract data.

## Resources

	# Tesseract and Pytesseract OCR engine
	- https://github.com/h/pytesseract
	- https://github.com/madmaze/pytesseract
	- https://github.com/tesseract-ocr/tesseract
	- tesseract training data
			- https://tesseract-ocr.github.io/tessdoc/Data-Files


	# Tesseract text localization and detection
	- https://pyimagesearch.com/2020/05/25/tesseract-ocr-text-localization-and-detection/?fbclid=IwAR2lVNBwSenvXEl0E6XQMZ8WUcKg9QPCeplUz2DBpIB_Eqx9kFcmIf40tys
	- https://nanonets.com/blog/ocr-with-tesseract/
	- https://medium.com/nanonets/a-comprehensive-guide-to-ocr-with-tesseract-opencv-and-python-fd42f69e8ca8


	# YOLO
	- https://towardsdatascience.com/the-practical-guide-for-object-detection-with-yolov5-algorithm-74c04aac4843 
	- https://stackoverflow.com/questions/61177652/does-every-occurence-of-an-object-needs-to-be-labelled-when-annotating-them-for 

	# Some good issues discussion
	- https://github.com/ultralytics/yolov5/issues/5851 
	- https://github.com/ultralytics/yolov5/issues/7562 
	
	# Rotated bounding boxes
	- https://ai.stackexchange.com/questions/9934/is-it-difficult-to-learn-the-rotated-bounding-box-for-a-rotated-object/9997#9997?newreg=944ec35aa4d3424bb314defbada5e988 
	- https://github.com/ultralytics/yolov3/issues/345 

	- improve ocr accuracy
		https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/ (NOTE: try this)
	
	# Additional reading 
	- https://pyimagesearch.com/start-here/
	- https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects 
	- https://medium.com/@vinay.dec26/yat-an-open-source-data-annotation-tool-for-yolo-8bb75bce1767 
	- https://www.molbiolcell.org/doi/10.1091/mbc.E20-02-0156
	
	# Image Labeling
	- https://foobar167.medium.com/open-source-free-software-for-image-segmentation-and-labeling-4b0332049878
	- https://www.robots.ox.ac.uk/~vgg/software/via/via.html 
	- https://makesense.ai
	- http://www.fexovi.com/sefexa.html 
	
	## Segmentation Models
	- https://ods.ai/projects/segmentation_models
	- https://github.com/qubvel/segmentation_models
	- https://github.com/qubvel/segmentation_models/blob/master/examples/binary%20segmentation%20(camvid).ipynb
	- https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb
	

	- https://github.com/ahmetozlu/signature_extractor
	- https://medium.com/analytics-vidhya/signature-recognition-using-opencv-2c99d878c66d
	
	# text clustering
	- https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/joe.2018.8282
	- https://www.geeksforgeeks.org/clustering-in-machine-learning/ 
	
	# specifically YOLO tutorials
	- https://appsilon.com/object-detection-yolo-algorithm
	- https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006
	
	# easy text detector
	- https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/ (NOTE: try this)
	- https://towardsdatascience.com/scene-text-detection-and-recognition-using-east-and-tesseract-6f07c249f5de
	- https://towardsdatascience.com/pytorch-scene-text-detection-and-recognition-by-craft-and-a-four-stage-network-ec814d39db05
	- https://github.com/janzd
	- https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet
	- 

	- https://pyimagesearch.com/2015/11/30/detecting-machine-readable-zones-in-passport-images/ (NOTE: check this out)

	# using prebuilt model
	- https://stackoverflow.com/questions/54821969/how-to-make-bounding-box-around-text-areas-in-an-image-even-if-text-is-skewed

	# install tesseract
	- https://techviewleo.com/how-to-install-tesseract-ocr-on-ubuntu/
	- https://linuxhint.com/install-tesseract-ocr-linux/
	- https://muthu.co/all-tesseract-ocr-options/

	

	# tesseract PSM (more detailed tutorials)
	- https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/
	- https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
	- understand Tesseract data
		https://stackoverflow.com/questions/61461520/does-anyone-knows-the-meaning-of-output-of-image-to-data-image-to-osd-methods-o


	# training custom Tesseract OCR model
	- https://towardsdatascience.com/train-a-custom-tesseract-ocr-model-as-an-alternative-to-google-vision-ocr-for-reading-childrens-db8ba41571c8
	- https://github.com/BloomTech-Labs/scribble-stadium-ds
	- https://pretius.com/blog/ocr-tesseract-training-data/
	- https://tesseract-ocr.github.io/tessdoc/tess4/TrainingTesseract-4.00.html
	- https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
	- https://www.statworx.com/en/content-hub/blog/fine-tuning-tesseract-ocr-for-german-invoices/

	# discussion forums
	- https://groups.google.com/g/tesseract-ocr/c/hE3B3EOwCXg

	# edge detection
	- https://gitlab.lrde.epita.fr/olena/pylene


	# TODO: after the model is build also try with Tesseract version 5

	# OpenCV Edge Detection
	- https://pyimagesearch.com/2021/10/27/automatically-ocring-receipts-and-scans/  # handles skewed situation
	- https://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python
	- https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html
	- https://learnopencv.com/edge-detection-using-opencv/
	- https://pyimagesearch.com/2021/05/12/opencv-edge-detection-cv2-canny/
	- https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/
	- https://pyimagesearch.com/2020/09/07/ocr-a-document-form-or-invoice-with-tesseract-opencv-and-python/


	##### OpenCV EAST Detector
	- https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
	- https://www.folio3.ai/blog/text-detection-by-using-opencv-and-east/
	- https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV
	- https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
	- https://pyimagesearch.com/2015/11/30/detecting-machine-readable-zones-in-passport-images/
	- https://github.com/cs-chan/Total-Text-Dataset
	- https://stackoverflow.com/questions/54821969/how-to-make-bounding-box-around-text-areas-in-an-image-even-if-text-is-skewed

	CRAFT: Character-Region Awareness For Text detection (try this)
		https://medium.com/technovators/scene-text-detection-in-python-with-east-and-craft-cbe03dda35d5 

	# EAST on custom data
	- https://github.com/argman/EAST
	- https://stackoverflow.com/questions/59710912/train-east-text-detector-on-custom-data
	- https://indiantechwarrior.com/text-detection-in-ocr-using-east/
	- https://github.com/indiantechwarrior/EAST

	# OpenCV, Keras and Tensorflow
	- for handwriting detection and OCR
		https://pyimagesearch.com/2020/08/24/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow/
		https://blog.devgenius.io/handwritten-text-recognition-using-convolutional-neural-networks-cnn-714d69d28c9a
	- 

	# DataSet
	- https://bgshih.github.io/cocotext/#h2-download
	- https://paperswithcode.com/paper/coco-text-dataset-and-benchmark-for-text


	# Semantic Segmentation
	- https://medium.com/technovators/semantic-segmentation-using-deeplabv3-ce68621e139e


	# removing lines in an image using pytesseract
	- https://stackoverflow.com/questions/48327567/removing-horizontal-underlines

	# setting up OCR server
	- https://realpython.com/setting-up-a-simple-ocr-server/

	# opencv color splitting
	- https://pyimagesearch.com/2021/04/28/opencv-color-spaces-cv2-cvtcolor/

	# improving text detection using GPUs
	- https://pyimagesearch.com/2022/03/14/improving-text-detection-speed-with-opencv-and-gpus/

	# opencv object detection
	- https://pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
	- https://pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/

	# correct skewness (image rotation) using opencv
	- https://pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
	- https://stackoverflow.com/questions/54821969/how-to-make-bounding-box-around-text-areas-in-an-image-even-if-text-is-skewed
	- 

	# blacklisting characters in pytesseract
	- https://nanonets.com/blog/ocr-with-tesseract/
	<!-- custom_config = r'-c tessedit_char_blacklist=0123456789 --psm 6' -->
	<!-- pytesseract.image_to_string(img, config=custom_config) -->

	# bank check ocr (not very detailed though, covers only numbers)
	- https://pyimagesearch.com/2017/07/24/bank-check-ocr-with-opencv-and-python-part-i/
	- https://pyimagesearch.com/2017/07/31/bank-check-ocr-opencv-python-part-ii/

	# credit card ocr
	- https://pyimagesearch.com/2017/07/17/credit-card-ocr-with-opencv-and-python/

	# regex price pattern
	pricePattern = r'([0-9]+\.[0-9]+)'

	# imutils package from pyimagesearch
	- https://pyimagesearch.com/2015/02/02/just-open-sourced-personal-imutils-package-series-opencv-convenience-functions/







	
	
	
## Data Source


## Data Preparation


### File Naming

Files are named according to the following way:

numerals_i - indicates the file that was downloaded from the internet
numerals_w - indicates the file that was provided by wellsfargo vendor
alph_i - indicates the file that is not a check
alph_w - indicates the file that is not a check image but other and was provided by wellsfargo


### Test Train Data Set



### Object Annotation

The fields we can annotate in a check image are: 
	item {
	  id: 1
	  name: 'IssueBank'
	}
	item {
	  id: 2
	  name: 'ReceiverName'
	}
	item {
	  id: 3
	  name: 'AcNo'
	}
	item {
	  id: 4
	  name: 'Amt'
	}
	item {
	  id: 5
	  name: 'ChqNo'
	}
	item {
	  id: 6
	  name: 'DateIss'
	}
	item {
	  id: 7
	  name: 'Sign'
	}
	
	
### Converting data to grayscale



	



