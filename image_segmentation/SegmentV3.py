import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import gaussian_filter
import time


class Processing:
    """Class for processing images."""

    def __init__(self, pic, depth):
        """Initialize the Processing class."""

        self._pic = pic
        self._depth = depth

    def calculate_threshold(self, bw):
        """Calculate the threshold based on the border of the image."""

        mean_number_sum = 0
        count = 0
        h, w = self._pic.shape

        # Get the inside area of the image excluding the border
        inside = self._pic[bw:-bw, bw:-bw]
        # Create a zero array with the same shape as the image
        zero_array = np.zeros((h, w))
        # Fill the zero array with the inside area
        zero_array[bw:-bw, bw:-bw] = inside
        borderless = zero_array
        # subtract the borderless image from the original image to get the border
        border = self._pic - borderless

        # go through the border and calculate the mean and max values
        for i in range(len(border)):
            row = border[i]
            for g in range(len(row)):
                number = row[g]
                if number > 0:
                    count += 1
                    mean_number_sum += number

        mean_threshold = (mean_number_sum / count * 0.95) + (np.max(border) * 0.05)
        max_threshold = np.max(border) * 1.2
        print("Mean number:", mean_number_sum / count)
        print("Max number:", np.max(border))
        print("mean threshold:", mean_threshold)
        print("Max threshold:", max_threshold)

        return mean_threshold, max_threshold

    def pic_devider(self):
        """Divide the image into four quadrants."""

        # Get the dimensions of the image
        h, w = self._pic.shape

        if h < 2 or w < 2:
            return [self._pic]

        # Ensure the image dimensions are even
        h2, w2 = h // 2, w // 2

        # Divide the image into four quadrants
        top_left = self._pic[:h2, :w2]
        top_right = self._pic[:h2, w2:]
        bottom_left = self._pic[h2:, :w2]
        bottom_right = self._pic[h2:, w2:]
        return [top_left, top_right, bottom_left, bottom_right]

    def check(self, mean_threshold, max_threshold, skip, first_layer_list, n):
        """Recursively segment the image based on the threshold."""

        # if the current depth exceeds the maximum depth, return
        if n >= self._depth:
            return
        # go through each image in the current layer
        for self._pic in first_layer_list[n]:
            # divide
            segments = self.pic_devider()
            # check the density of each segment
            for sec in segments:
                density = np.sum(sec) / (sec.shape[0] * sec.shape[1])
                if n >= skip and (
                    density <= mean_threshold or np.max(sec) <= max_threshold
                ):
                    sec *= 0  # set to zero if below threshold
                    sec += (n + 1) * 400  # set to zero if below threshold
                else:
                    # if the density is above the threshold,
                    # add the segment to the next layer
                    first_layer_list[n + 1].append(sec)

        # Recursively check the next depth level
        self.check(mean_threshold, max_threshold, skip, first_layer_list, n + 1)


def plot_image(image, title="Image", xlabel="x", ylabel="y"):
    """Plot an image with a colorbar."""

    plt.imshow(image, origin="lower")
    plt.colorbar(label="Amplitude")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def main(depth, skip, border_width, sigma):
    """combine everything in one function."""

    # Load the beam and background images and process them
    ####################################################################################
    beam_file = open("tds\\OTRC.2560.T3-20250309_131230_481.pcl", "rb")
    back_file = open("tds\\OTRA.473.B2D-20250309_111945_501.pcl", "rb")
    beam = pickle.load(beam_file)
    back = pickle.load(back_file)

    # subtract the background from the beam
    # and add a constant to avoid negative values
    subtracted_img = beam.astype(float) - back.astype(float)  # [200:1800, 400:1750]TBD
    subtracted_img[subtracted_img < 0] = 0

    # Apply a Gaussian filter to the subtracted image
    blurred_img = gaussian_filter(subtracted_img, sigma)
    ####################################################################################

    # Plot the original image
    plot_image(subtracted_img, title="subtracted image", xlabel="x", ylabel="y")
    # plot the blurred image
    plot_image(blurred_img, title="Blurred Image", xlabel="x", ylabel="y")

    # Initialize a list to hold the segmented images at each depth
    # each element of the list corresponds to a depth level
    first_layer_list = [[] for _ in range(depth + 1)]

    # Start the timer for the segmentation process
    start = time.time()

    pic = blurred_img  # Initial image to be segmented
    first_layer_list[0].append(pic)  # Add the initial image to the first layer list

    processed = Processing(pic, depth)  # Initialize the Processing class

    # Segment the image
    ####################################################################################
    # Calculate the thresholds based on the border of the image
    thresholds = processed.calculate_threshold(border_width)
    mean_threshold = thresholds[0]  # Get the mean threshold value
    max_threshold = thresholds[1]  # Get the maximum threshold value

    threshold_time = time.time() - start

    # Start the segmentation process
    processed.check(mean_threshold, max_threshold, skip, first_layer_list, n=0)
    ####################################################################################

    segmentation_time = time.time() - start - threshold_time

    # duration of the threshold calculation and segmentation process
    print(
        "threshold calculation time:",
        threshold_time,
        "segmentation time",
        segmentation_time,
    )

    processed_img = pic
    # Plotting the final segmented image
    plot_image(processed_img, title="Processed Image", xlabel="x", ylabel="y")


if __name__ == "__main__":
    depth = 10  # Maximum depth of segmentation
    skip = 0  # Skip depths for segmentation
    border_width = 1  # Width of the border for threshold calculation
    sigma = 2  # Sigma for Gaussian filter
    main(depth, skip, border_width, sigma)
