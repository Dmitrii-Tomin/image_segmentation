import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter
import time


class Processing:
    """Class for processing images."""

    def __init__(self, pic):
        """Initialize the Processing class."""

        self._pic = pic

    def calculate_threshold(self, size):
        """Calculate the threshold based on the border of the image."""

        h, w = self._pic.shape
        prev_max = 0

        tl = self._pic[:size, :size]
        tr = self._pic[:size, w - size :]
        bl = self._pic[h - size :, :size]
        br = self._pic[h - size :, w - size :]

        corners = [tl, tr, bl, br]

        for corner in corners:
            if np.max(corner) > prev_max:
                maximum = np.max(corner)
                prev_max = maximum

        max_threshold = maximum * 1.5

        return max_threshold

    def check(self, max_threshold):
        """Check the image and apply the thresholds."""

        output = np.where(self._pic < max_threshold, np.nan, self._pic)

        return output


def plot_image(image, title="Image", xlabel="x", ylabel="y"):
    """Plot an image with a colorbar."""

    plt.imshow(image, origin="lower")
    plt.colorbar(label="Amplitude")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def main(size, sigma):
    """combine everything in one function."""

    # Load the beam and background images and process them
    ####################################################################################
    beam_file = open(
        "image_segmentation\\tds\\OTRC.2560.T3-20250309_131230_481.pcl", "rb"
    )
    back_file = open(
        "image_segmentation\\tds\\OTRA.473.B2D-20250309_111945_501.pcl", "rb"
    )
    beam = pickle.load(beam_file)
    back = pickle.load(back_file)

    # Start the timer
    start = time.time()

    # subtract the background from the beam
    # and add a constant to avoid negative values
    subtracted_img = beam.astype(float) - back.astype(float)  # [200:1800, 400:1750]TBD
    subtracted_img[subtracted_img < 0] = 0

    # Apply a Gaussian filter to the subtracted image
    blurred_img = gaussian_filter(subtracted_img, sigma)
    ####################################################################################

    pic = blurred_img.copy()  # Initial image to be segmented

    processed = Processing(pic)  # Initialize the Processing class

    initialization_time = time.time() - start

    # Segment the image
    ####################################################################################
    # Calculate the threshold
    max_threshold = processed.calculate_threshold(size)

    threshold_time = time.time() - start - initialization_time

    # Start the segmentation process
    output = processed.check(max_threshold)
    ####################################################################################

    segmentation_time = time.time() - start - initialization_time - threshold_time

    # duration of the segmentation process
    print(
        "initialization time:",
        initialization_time,
        "threshold calculation time:",
        threshold_time,
        "segmentation time:",
        segmentation_time,
        "total time:",
        time.time() - start,
    )

    processed_img = output

    # Plot the original image
    plot_image(subtracted_img, title="subtracted image", xlabel="x", ylabel="y")
    # plot the blurred image
    plot_image(blurred_img, title="Blurred Image", xlabel="x", ylabel="y")
    # Plotting the final segmented image
    plot_image(processed_img, title="Processed Image", xlabel="x", ylabel="y")


if __name__ == "__main__":
    size = 100  # Size for threshold calculation
    sigma = 2  # Sigma for Gaussian filter
    main(size, sigma)
