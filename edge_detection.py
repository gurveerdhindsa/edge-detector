"""edge_detection.py"""
__author__ = "Gurveer Dhindsa"

from PIL import Image
from pylab import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import ceil
import cv2

"""
Creates a gaussian filter

Args:
    image - the image being filtered
    sigma - an integer that determines the intensity of blurring

Returns:
    gaussian_filter - a filter that can be convoluted to image to smooth it
"""
def gaussian(image, sigma):
    # Convert image to numpy array
    image = np.asarray(image)

    # Compute the size of filter
    filter_size = 2 * ceil(3 * sigma) + 1

    # Allocate space for gaussian filter array
    kernel = np.zeros((filter_size, filter_size), dtype=float)

    # Iterate the allocated array...
    for x in range(-filter_size, filter_size):
        for y in range(-filter_size, filter_size):
            # Use gaussian formula to fill in a filter pixel value
            kernel[x, y] = (1 / (2 * np.pi * (sigma ** 2))) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Return the filter
    return kernel

"""
Convolutes image with a kernel

Args:
    image - the image being filtered
    kernel - a filter which is being applied to the image

Returns:
    result - an image with a kernel filter applied to it
"""
def convolution(image, kernel):
    # Grah the image dimensions
    imageHeight, imageWidth = image.shape
    # Grab the kernel dimensions
    kernelHeight, kernelWidth = kernel.shape

    # Allocate memory for output image, so we must consider the padding
    # I chose to use 'same' convolution because I want the output image to be the same size of input image
    # In general, the equation to calculate padding is: p = (k-1)//2 (where p=padding, k=kernel)
    padHeight = (kernelHeight - 1) // 2
    padWidth = (kernelWidth - 1) // 2

    # The result image (post-convolution) will be the same dimensions of the input image
    result = np.zeros(image.shape, dtype=float)

    # Apply padding to top, bottom, right and left of image, while allocating space for original image
    imagePadded = np.zeros((imageHeight + padHeight + padHeight, imageWidth + padWidth + padWidth), dtype=float)

    # Grab the padded image dimensions
    imagePaddedHeight, imagePaddedWidth = imagePadded.shape

    # Copy image into center
    imagePadded[padHeight:imageHeight + padHeight, padWidth:imageWidth + padWidth] = image

    # Iterate the original image pixel by pixel...
    for x in range(imageHeight):
        for y in range(imageWidth):
            # Multiply the padded image with kernel filter then take sum of all pixels to create a single pixel
            # The single pixel will be added to the result image
            result[x,y] = (np.matmul(imagePadded[x:kernelWidth + x, y:kernelHeight + y], kernel)).sum()

    # Return the convoluted image with kernel applied
    return result

"""
(Required function)
This function accepts a grayscaled image (img0) and a sigma value and returns the image with edges detected.

Args:
    img0 - the original image
    sigma - an integer that determines intensity of blurring

Returns:
    magnitude - the original image with edges detected
"""
def myEdgeFilter(img0, sigma):
    # Get the gaussian filter kernel
    kernel = gaussian(img0, sigma)

    # Convolute original image with kernel filter and get the smoothed image
    blurred = convolution(img0, kernel)

    # plt.imshow(blurred, cmap='gray')
    # plt.title("Smoothed Image")
    # plt.show()

    # Pre-defined sobel convolution kernels (x and y)
    sobelKernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    sobelKernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Obtain convoluted X and Y direction images with sobel kernel applied
    imgx = convolution(img0, sobelKernelX)
    imgy = convolution(img0, sobelKernelY)

    # Print the image gradients in both directions
    # print ('Image gradient in X-direction (imgx): \n', imgx)
    # print('Image gradient in Y-direction (imgy): \n', imgy)

    # plt.imshow(blurred, cmap='gray')
    # plt.title("Smoothed Image")
    # plt.show()

    # Calculate the gradient magnitude
    magnitude = np.sqrt(np.square(imgx) + np.square(imgy))

    # Calculate the gradient direction, then convert to degrees (easier to deal with)
    direction = np.arctan2(imgy, imgx)
    direction = np.rad2deg(direction)

    return magnitude, direction


def nonMaxSupression (magnitude, direction):
    # Array of possible gradient angles
    angles = np.array([0, 45, 90, 135])

    magnitudeHeight, magnitudeWidth = magnitude.shape

    # The result image (post non-maximum supression) will be the same dimensions of the input image
    result = np.zeros(magnitude.shape, dtype=float)

    # Iterate the magnitude image pixel by pixel...
    # Since we are looking at neighboring pixels, we must not fully iterate the range. Instead, start at 1,1 and iterate until 1 pixel is left in both x and y axis.
    for x in range(1, magnitudeHeight-1):
        for y in range(1, magnitudeWidth-1):
            # Look at the current pixel in gradient direction
            directionPixel = direction[x,y]

            # Determine which of the 4 cases the pixel is closest to
            closestValue = angles[np.abs(angles - directionPixel).argmin()]

            # Closest case is 0 degrees
            if (closestValue == 0):
                neighbor1 = magnitude[x+1, y]
                neighbor2 = magnitude[x-1, y]

            # Closest case is 45 degrees
            elif (closestValue == 45):
                neighbor1 = magnitude[x+1, y+1]
                neighbor2 = magnitude[x-1, y-1]

            # Closest case is 90 degrees
            elif (closestValue == 90):
                neighbor1 = magnitude[x, y+1]
                neighbor2 = magnitude[x, y-1]

            # Closest case is 135 degrees
            else:
                neighbor1 = magnitude[x+1 , y-1]
                neighbor2 = magnitude[x-1, y+1]

            # If either neighbors have a larger gradient magnitude, then set pixel to 0.
            # Since I created an empty array with non 0's, only copy the same pixel value from magnitude IF the neighbors are smaller than the gradient magnitude
            if (neighbor1 <= magnitude[x,y] and neighbor2 <= magnitude[x,y]):
                result[x,y] = magnitude[x,y] # Copy over the pixel from magnitude

    return result

"""
Determines if image is grayscale
(Helper function)

Args:
    image - the image being examined

Returns:
    A boolean value whether or not the image is already grayscale
"""
def isImageGrayscale(image):
    image = Image.fromarray(image) # Convert array back to image
    width, height = image.size # Grab the image dimensins

    # Iterate the image pixel by pixel...
    for w in range(width):
        for h in range(height):
            # Grab the RGB values
            r, g, b = image.getpixel((w, h))
            # If they are individually different, then we know it CANNOT be a grayscale image
            if r != g != b: 
                return False
    # If we got here, then the image is grayscale
    return True

"""
Main function
"""
def main():
    # Read image from root project directory
    # I chose to use my own image (homer_simpson.jpg)
    image = cv2.imread("images/homer_simpson.jpg")

    # Pre-defined sigma value
    sigma = 4

    # Convert image to grayscale
    if not isImageGrayscale(image):
        print("Coloured image detected, attempting to grayscale...")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Image successfully grayscaled!")

    magnitude, direction = myEdgeFilter(image, sigma)
    
    postNonMaxSupressionImage = nonMaxSupression(magnitude, direction)

    # Display the edge magnitude image & save under results
    plt.imshow(magnitude.astype(int), cmap='gray')
    plt.axis("off")
    plt.title("Gradient Magnitude Result")
    plt.savefig('./results/gradient_magnitude_result.png')
    plt.show()

    # Display the direction image & save under results
    plt.imshow(direction.astype(int), cmap='gray')
    plt.axis("off")
    plt.title("Gradient Direction Result")
    plt.savefig('./results/gradient_direction_result.png')
    plt.show()

    # Display the edge magnitude image & save under results
    plt.imshow(postNonMaxSupressionImage.astype(int), cmap='gray')
    plt.axis("off")
    plt.title("Gradient Magnitude (Post Non-Max Supression) Result")
    plt.savefig('./results/non_max_supression_result.png')
    plt.show()
    

if __name__ == "__main__":
    main()
