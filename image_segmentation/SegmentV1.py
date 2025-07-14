import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter
import time

# Load the beam and background data
beam_file = open("tds\\OTRC.2560.T3-20250309_131230_481.pcl", "rb")
back_file = open("tds\\OTRA.473.B2D-20250309_111945_501.pcl", "rb")
beam = pickle.load(beam_file)
back = pickle.load(back_file)

# subtract the background from the beam
# and add a constant to avoid negative values
subtracted_img = beam.astype(float) - back.astype(float)
subtracted_img[subtracted_img < 0] = 0

# Apply a Gaussian filter to the subtracted image
blurred_img = gaussian_filter(subtracted_img, sigma=2)

# Plot the original image
plt.imshow(subtracted_img, origin="lower")
plt.colorbar(label="Amplitude")
plt.title("original Image")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# plot the blurred image
plt.imshow(blurred_img, origin="lower")
plt.colorbar(label="Amplitude")
plt.title("Blurred Image")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

pic = blurred_img  # Initial image to be segmented
threshold = 10  # 1164 # 1230 # Threshold for segmentation
depth = 10  # Maximum depth of segmentation
n = 0  # Current depth of segmentation

# Initialize a list to hold the segmented images at each depth
# each element of the list corresponds to a depth level
first_layer_list = [[] for _ in range(depth + 1)]

# Start the timer for the segmentation process
start = time.time()

# Add the initial image to the first layer list
first_layer_list[n].append(pic)


def calulate_threshold(pic):
    """Calculate the threshold based on the first 100x100 pixels of the image."""
    return (np.max(pic[:50, :50]) + np.mean(pic[:50, :50])) / 2


def pic_devider(pic):
    """Divide the image into four quadrants."""
    # Get the dimensions of the image
    h, w = pic.shape

    # Ensure the image dimensions are even
    h2, w2 = h // 2, w // 2

    # Divide the image into four quadrants
    top_left = pic[:h2, :w2]
    top_right = pic[:h2, w2:]
    bottom_left = pic[h2:, :w2]
    bottom_right = pic[h2:, w2:]
    return [top_left, top_right, bottom_left, bottom_right]


def check(n, depth, first_layer_list):
    """Recursively segment the image based on the threshold."""
    # if the current depth exceeds the maximum depth, return
    if n >= depth:
        return

    # go through each image in the current layer
    for pic in first_layer_list[n]:
        # divide
        segments = pic_devider(pic)
        # check the density of each segment
        for sec in segments:
            density = np.sum(sec) / (sec.shape[0] * sec.shape[1])
            if density <= threshold:
                sec += 140 * (n + 1)  # set to zero if below threshold
            else:
                # if the density is above the threshold,
                # add the segment to the next layer
                first_layer_list[n + 1].append(sec)

    # Recursively check the next depth level
    check(n + 1, depth, first_layer_list)


# Start the segmentation process
check(n, depth, first_layer_list)

# duration of the segmentation process
print("Time taken:", time.time() - start)

# Plotting the final segmented image
plt.imshow(blurred_img, aspect="auto", origin="lower")
plt.colorbar(label="Amplitude")
plt.title("2D Gaussian (σₓ ≠ σᵧ)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
