# CheckImageScanning
Application to scan the check image and extract data.

The overall application workflow is split into:

- Read image data from a source, or give an input image.
- Scan the image using EAST, and identify potential text instances in the image.
- OCR the text instance using Tesseract.
- Clean the OCR text.
- Label the OCR text.
