[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Image Segmentation

This project implements a recursive image segmentation algorithm for extracting the Region of Interest (ROI) of a Gaussian beam from a 2D image, typically obtained by subtracting a background signal from a measurement. The segmentation is based on intensity thresholds and works by dividing the image into smaller sections until only the significant parts of the beam remain.

<img width="1177" height="839" alt="Figure_5" src="https://github.com/user-attachments/assets/3a25f5cb-4143-4b45-8938-cba248bf774b" />

# Results

   | Original Image                   | Segmented (Colored by Depth)                   | Segmented (RONI = NaN)                     |
   | -------------------------------- | ---------------------------------------------- | ------------------------------------------ |
<img width="6000" height="3375" alt="Untitled design (9)" src="https://github.com/user-attachments/assets/4c38b81d-f785-4e46-9dd3-64729727cf59" />

