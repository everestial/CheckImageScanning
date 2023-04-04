# List of requirements

## Tesseract

    $ tesseract --version
    tesseract 4.1.1
    leptonica-1.82.0
      libgif 5.1.9 : libjpeg 8d (libjpeg-turbo 2.1.1) : libpng 1.6.37 : libtiff 4.3.0 : zlib 1.2.11 : libwebp 1.2.2 : libopenjp2 2.4.0
    Found AVX2
    Found AVX
    Found FMA
    Found SSE
    Found libarchive 3.6.0 zlib/1.2.11 liblzma/5.2.5 bz2lib/1.0.8 liblz4/1.9.3 libzstd/1.4.8


## Python packages

Available in the file `requirements-main-app.txt`.

    $ cat textOCR-mainApp/requirements.txt
    
    # if updating the requirements do
    $ pip freeze > requirements-main-app.txt

