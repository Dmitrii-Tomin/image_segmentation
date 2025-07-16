[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Image Segmentation

This project implements a recursive image segmentation algorithm for extracting the Region of Interest (ROI) of a Gaussian beam from a 2D image, typically obtained by subtracting a background signal from a measurement. The segmentation is based on intensity thresholds and works by dividing the image into smaller sections until only the significant parts of the beam remain.

<img width="1177" height="839" alt="Figure_5" src="https://github.com/user-attachments/assets/3a25f5cb-4143-4b45-8938-cba248bf774b" />

# How to create the conda environment

<img width="6000" height="3375" alt="Untitled design (6)" src="https://github.com/user-attachments/assets/a30ce52a-1299-4dd8-b5bd-2613c63d2a43" />
<img width="6000" height="3375" alt="Untitled design (7)" src="https://github.com/user-attachments/assets/cd1852fe-fca3-4369-b9dd-9dda933732b4" />

`conda env create -f condaenv.yaml`
