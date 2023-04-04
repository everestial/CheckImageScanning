

Sources:

    - https://tesseract-ocr.github.io/tessdoc/#traineddata-files 
    - https://tesseract-ocr.github.io/tessdoc/tess4/TrainingTesseract-4.00.html 

### Setup Training

Setup virtual enviornment (separately for training)

install requirements
  pip install -r requirements.txt

MAKE
  make leptonica tesseract

  make tesseract-langdata


???
  $ make
  $ sudo make install
  $ sudo ldconfig

  Documentation will not be built because asciidoc or xsltproc is missing.

  Training tools can be built and installed with:

  $ make training
  $ sudo make training-install
