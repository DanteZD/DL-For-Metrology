import numpy as np
import cv2 as cv
import os

directory = 'data-4/'
number = 0

#median or gaussian
blur_type = 'median'
#strenght of blur, odd number (19)
blur_para = 21

#width and height of adaptive threshold region (255-533)
# depends on hole size -> bigger holes look at a bigger area
adapt_thresh_para = 1111

#Kernal size noise removal 5 - 43 is goed geweest
noise_kernel_size = 33

#values between 0 and 1 (0.7 default)
dist_transform_para = 0.42

#Blur the original image to remove noise (such as crosshair) before applying watershed (5)
watershed_blur = 7


for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        # Loading the image
        img = cv.imread(f)

        # Grayscale and Blurred!
        assert img is not None, "file could not be read, check with os.path.exists()"
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # blur the image to remove noise
        if blur_type == 'median':
            gray = cv.medianBlur(gray, blur_para)
        elif blur_type == "gaussian":
            gray = cv.GaussianBlur(gray, (blur_para, blur_para), 0)

        # adaptive thresholding for binary image.
        # Gaussian 333, 0
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, adapt_thresh_para, 0)

        # ******** noise removal ************
        kernel = np.ones((noise_kernel_size, noise_kernel_size), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        # sure background area
        sure_bg = cv.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)

        # SuperSmall 0.1, Small 0.3, medium 0.5, large 0.7
        ret, sure_fg = cv.threshold(dist_transform, dist_transform_para * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        # Blur the image which watershed uses to remove some noise
        # This helps with: The lens flare and the crosshair affecting the segmentations
        img_blurred = cv.medianBlur(img, watershed_blur)

        # Watershed
        markers = cv.watershed(img_blurred, markers)

        #Save original image
        cv.imwrite(directory + 'original/img' + str(number) + '.jpg', img)

        # Resulting segmentation on image
        img[markers == -1] = [255, 0, 0]

        # Save segmentation mask
        cv.imwrite(directory + 'mask/' + 'mask' + str(number) + '.jpg', markers)

        # Save original image with segmentation lines on
        cv.imwrite(directory + 'original_with_mask/result' + str(number) + '.jpg', img)
        number += 1