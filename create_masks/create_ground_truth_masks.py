"""
Create lens or sieve ground truth segmentation masks.
Using a json file with annotated regions.
"""

import numpy as np
import cv2 as cv
from PIL import Image
import os
import json

# Select dataset
dataset = "data-3/"
# Lists of all images in the dataset and their corresponding json annotations.
input_images = os.listdir("images/" + dataset)
input_annotations = os.listdir("annotations/" + dataset)
image_dir = os.path.join("images/", dataset)
annotation_dir = os.path.join("annotations/", dataset)

for input_image, input_annotation in zip(reversed(input_images), reversed(input_annotations)):
    img_path = os.path.join(image_dir, input_image)
    annotation_path = os.path.join(annotation_dir, input_annotation)

    # read an individual image
    img = np.array(Image.open(img_path))

    # get all corresponding annotated segments of the image
    f = open(annotation_path)
    image_segments = json.load(f)

    # Create background of the mask with a shape corresponding to the input image
    mask = np.zeros([img.shape[0], img.shape[1]])

    # fill in the area spanned by the segments with their corresponding class color
    for segment in image_segments:
        label = segment['label']
        area = np.array(segment['points']).astype(int)

        # FOR SCOPE SEGMENTS
        if label == 'scope':
            mask = cv.fillPoly(mask, pts = [area], color = 255)

        # FOR SIEVE SEGMENTS
        #     if label == 'hole':
        #         mask = cv.fillPoly(mask, pts = [area], color = 1)
        #     elif label == 'debris':
        #         mask = cv.fillPoly(mask, pts = [area], color = 2)

    #save masks
    cv.imwrite("masks/" + dataset + input_image.replace('.jpg', '_mask.png'), mask)
