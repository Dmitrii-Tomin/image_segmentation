import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter
import math


class Processing:
    """Class for processing images."""

    def __init__(self, pic, depth):
        """Initialize the Processing class."""

        self._pic = pic
        self._depth = depth

    def calculate_threshold(self, size, thresh_mult):
        """Calculate the threshold based on the border of the image."""

        h, w = self._pic.shape
        averages = 0
        prev_max = 0
        maximum = 0

        tl = self._pic[:size, :size]
        tr = self._pic[:size, w - size :]
        bl = self._pic[h - size :, :size]
        br = self._pic[h - size :, w - size :]

        corners = [tl, tr, bl, br]

        # Calculate the average of each corner
        for corner in corners:
            averages += np.mean(corner)
            # Get the highest number of all the corners
            if np.max(corner) > prev_max:
                maximum = corner.max()
                prev_max = maximum

        average = averages / 4

        mean_threshold = average
        max_threshold = maximum * thresh_mult

        return mean_threshold, max_threshold

    def pic_divider(self, pic):
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

    def check(self, mean_threshold, max_threshold, skip, layers_list, mval, n):
        """Recursively segment the image based on the threshold."""

        # if the current depth exceeds the maximum depth, return
        if n >= self._depth:
            return
        # go through each image in the current layer
        for tile in layers_list[n]:
            # Divide into quadrants
            segments = self.pic_divider(tile)
            # check the density of each segment
            for sec in segments:
                density = np.sum(sec) / (sec.shape[0] * sec.shape[1])
                if n >= skip and (
                    density <= mean_threshold or sec.max() <= max_threshold
                ):
                    sec.fill(mval)  # set to NaN if below threshold

                    # or for colored visualization:
                    # sec *= 0  # set to zero if below threshold
                    # sec += (n + 1) * 400  # depending on the depth add value
                else:
                    # add the segment to the next layer
                    layers_list[n + 1].append(sec)

        # Recursively check the next depth level
        self.check(mean_threshold, max_threshold, skip, layers_list, mval, n + 1)


class Segment:
    """Encapsulates image segmentation using recursive quadrant-based filtering."""

    def __init__(self, initial_img, depth, skip, sigma, thresh_win, thresh_mult, mval):
        """Initialize the SegmentApp with parameters."""
        self._initial_img = initial_img
        self._depth = depth
        self._skip = skip
        self._thresh_win = thresh_win
        self._sigma = sigma
        self._thresh_mult = thresh_mult
        self._mval = mval

    @property
    def blurred_image(self) -> np.ndarray:
        """Returns a Gaussian-blurred version of the initial image."""
        blurred = gaussian_filter(self._initial_img, self._sigma)
        return blurred

    def start_segment(self):
        """Start the segmentation application."""

        original_img = self._initial_img.copy()

        # Apply a Gaussian filter to the subtracted image
        blurred_img = self.blurred_image
        pic = blurred_img.copy()  # Initial image to be segmented

        """calculate the maximum depth based on the image size"""
        ################################################################################
        h, w = pic.shape
        shortest_side = min(h, w)
        max_depth = math.floor(math.log2(shortest_side / 2)) + 1
        if self._depth > max_depth:
            print(
                f"Warning: The specified depth {self._depth} "
                f"exceeds the maximum depth {max_depth} "
                f"for the image size {h}x{w}. "
                f"Adjusting depth to {max_depth}."
            )
            self._depth = max_depth
        ################################################################################

        # Initialize a list to hold the segmented images at each depth
        # each element of the list corresponds to a depth level
        layers_list = [[] for _ in range(self._depth + 1)]
        layers_list[0].append(pic)  # Add the initial image to the first layer list

        processed = Processing(pic, self._depth)

        """Segment the image"""
        ################################################################################
        # Calculate the thresholds based on the border of the image
        thresholds = processed.calculate_threshold(self._thresh_win, self._thresh_mult)
        mean_threshold = thresholds[0]
        max_threshold = thresholds[1]

        # Start the segmentation process
        processed.check(
            mean_threshold, max_threshold, self._skip, layers_list, self._mval, n=0
        )
        ################################################################################

        if np.isnan(self._mval):
            mask = np.isnan(pic)  # Create a mask for NaN values
        else:
            mask = pic == mval

        original_img[mask] = mval  # Set NaN values in the original image

        processed_img = original_img
        # processed_img = pic # Use for colored output

        return processed_img


def plot_image(image, title="Image", xlabel="x", ylabel="y"):
    """Plot an image with a colorbar."""

    plt.imshow(image, origin="lower")
    plt.colorbar(label="Amplitude")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == "__main__":
    depth = 10  # Maximum depth of segmentation
    skip = 0  # Skip depths for segmentation
    sigma = 2  # Sigma for Gaussian filter
    thresh_win = 100  # Size of the boxes for threshold calculation
    thresh_mult = 1.5  # Threshold multiplier
    mval = np.nan  # value used to mask out RONI

    """Load the beam and background images and process them"""
    ####################################################################################
    beam_file = open("tds\\OTRC.2560.T3-20250309_131230_481.pcl", "rb")
    back_file = open("tds\\OTRA.473.B2D-20250309_111945_501.pcl", "rb")
    beam = pickle.load(beam_file)
    back = pickle.load(back_file)

    # subtract the background from the beam
    # and add a constant to avoid negative values
    initial_img = beam.astype(float) - back.astype(float)
    initial_img[initial_img < 0] = 0
    ####################################################################################

    # Create an instance of SegmentApp with the initial image and parameters
    app = Segment(initial_img, depth, skip, sigma, thresh_win, thresh_mult, mval)

    blurred_img = app.blurred_image
    processed_img = app.start_segment()

    # Plotting the images
    plot_image(initial_img, title="Image", xlabel="x", ylabel="y")
    plot_image(blurred_img, title="Image", xlabel="x", ylabel="y")
    plot_image(processed_img, title="Image", xlabel="x", ylabel="y")
