[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Beam Image Segmentation

This project implements a recursive image segmentation algorithm for extracting the Region of Interest (ROI) of a Gaussian beam from a 2D image, typically obtained by subtracting a background signal from a measurement. The segmentation is based on intensity thresholds and works by dividing the image into smaller sections until only the significant parts of the beam remain.

<img width="1177" height="839" alt="Figure_5" src="https://github.com/user-attachments/assets/3a25f5cb-4143-4b45-8938-cba248bf774b" />

## Features

- **Recursive Quad-Tree Segmentation**
- **Gaussian Blur Preprocessing**
- **Adaptive Local Thresholding** (Mean and Max)
- **Automatically masks segments with low average intensity or low maximum values**
- **Adjusts if depth is too large for image size**


## Usage

```python
Segment(initial_img, depth, skip, sigma, thresh_win, thresh_mult, mval)
```

`initial_img` : ndarray
 The input 2D image array to be segmented.

`depth` : int
 Maximum recursion depth for segmentation.

`skip` : int
 Number of depth levels to skip before applying threshold masking.

`sigma` : float
 Sigma value for Gaussian blur applied before segmentation.

`thresh_win` : int
 Size of the box regions used for threshold calculation.

`thresh_mult` : float
 Multiplier applied to the maximum threshold for sensitivity adjustment.

`mval` : float or np.nan
 The value used to mask out low-intensity segments (e.g., np.nan or 0).


### Example

```python
import numpy as np
from image_segmentation import Segment
import matplotlib.pyplot as plt

# Load or prepare your initial image as a 2D NumPy array
initial_img = np.load('your_image.npy')  # example

# Create a Segment instance
segmenter = Segment(
    initial_img=initial_img,
    depth=10,
    skip=0,
    sigma=2,
    thresh_win=100,
    thresh_mult=1.5,
    mval=np.nan
)

# Run segmentation
processed_img = segmenter.start_segment()

# Use matplotlib to visualize the result
plt.imshow(processed_img, origin='lower')
plt.colorbar()
plt.title("Segmented Image")
plt.show()

```


## Installation

**Install using:**
```
pip install git+https://github.com/Dmitrii-Tomin/image_segmentation.git
```
**or clone repository and install:**
```
git clone https://github.com/Dmitrii-Tomin/image_segmentation.git
cd image_segmentation
pip install .
```

### Create conda evironment

`conda env create -f condaenv.yaml`


# Results

   | Original Image                   | Segmented (Colored by Depth)                   | Segmented (RONI = NaN)                     |
   | -------------------------------- | ---------------------------------------------- | ------------------------------------------ |
   
<img width="6000" height="3375" alt="Untitled design (9)" src="https://github.com/user-attachments/assets/4c38b81d-f785-4e46-9dd3-64729727cf59" />

