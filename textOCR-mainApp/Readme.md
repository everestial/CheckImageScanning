

### Usage:

```bash
$ python text_scan_main.py -h
usage: text_scan_main.py [-h] [-o OUTFILE] inFile

Process input and output files.

positional arguments:
  inFile                Input file path

options:
  -h, --help            show this help message and exit
  -o OUTFILE, --outFile OUTFILE
                        Output file path (optional)
```

**Specific example:**

```
$ python text_scan_main.py images/012.jpg 
Image Values:
 original_height, original_width, ratio_width_O2N, ratio_height_O2N, new_height, new_width 
 584 1280 4.0 1.825 320 320
Image read from an existing OpenCV object (numpy array).
[INFO] loading EAST text detector...
[INFO] text detection took 0.391446 seconds

Completed OCR of the image images/012.jpg
Output data written in file with prefix images/012
```