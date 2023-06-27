import cv2 as cv
import numpy as np
import json
import albumentations as A
import csv


class Segment():
    def __init__(self):
        self.contour = None
        self.area = None
        self.width = None
        self.height = None
        self.x = None
        self.y = None
        self.error = None


def remove_background(segments, scope_mask):
    remaining_mask_segments = []

    # Iterate over the mask segments and check if their contour lies ether completely or partly outside the scope contour
    for i, seg in enumerate(segments):
        # Create a binary blank image with the same size as the scope mask image
        segment_mask = np.zeros(scope_mask.shape[:2], dtype=np.uint8)
        # Add the segment of interest to the segment mask
        cv.drawContours(segment_mask, [seg], 0, (255), thickness=cv.FILLED)

        # Compute the difference between the segment maks and the scope mask
        dif = cv.subtract(segment_mask, scope_mask)

        # If the difference between the 2 masks is bigger than 0 at some pixel location
        # It means the pixel of the segment at that location falls outside the scope mask
        # Check if the difference is non-zero
        # And remove (skip) segment if this is the case
        if np.any(dif > 0):
            continue

        # Else add it to the remaining mask segments that fall inside the scope region
        remaining_mask_segments.append(seg)

    return remaining_mask_segments


def remove_segments(segments, scope_mask, margin):
    # Remove all (remaining) sieve segments that fall partially or completely outside the scaled scope
    # with a margin of 10%
    segments_of_interest = []
    # Removing edge holes that are cut-off by the scaled scope as these provide no valid info
    for i, seg in enumerate(segments):
        # Create a binary blank image with the same size as the mask image
        segment_mask = np.zeros(scope_mask.shape[:2], dtype=np.uint8)
        # Add the segment of interest to the segment mask
        cv.drawContours(segment_mask, [seg], 0, (255), thickness=cv.FILLED)

        # Compute the difference between the segment maks and the scope mask
        dif = cv.subtract(segment_mask, scope_mask)

        # number of pixels outside the scaled scope.
        pixels_outside = np.sum(dif > 0)
        # total number of pixels of the segment
        total_segment_pixels = np.sum(segment_mask > 0)

        # Remove (skip) segments that have more than 5% of their pixels fall outside the scope area
        if pixels_outside / total_segment_pixels > margin:
            continue

        segments_of_interest.append(seg)

    return segments_of_interest


def fuse_segments(segments, mask, new_mask, threshold):
    for seg in segments:
        # Create a blank mask for the current segment
        count_mask = np.zeros_like(mask)
        # Fill it with the contour of the segment
        cv.fillPoly(count_mask, [seg], color=(255, 255, 255))

        # The region a single segment in the original mask
        roi = cv.bitwise_and(mask, count_mask)

        # Count the number of pixels per class in the segment
        hole_pixels = np.count_nonzero(roi == 127)
        debris_pixels = np.count_nonzero(roi == 255)
        total_pixels = hole_pixels + debris_pixels

        # If atleast 20% of the pixels are off the debris class
        if debris_pixels >= total_pixels * threshold:
            # Set all pixels of the segment to 255
            new_mask[np.where(roi != 0)] = 255
        else:
            new_mask[np.where(roi != 0)] = 127

    return new_mask


def calc_avg_area(contours):
    if len(contours) <= 0:
        return 0

    areas = np.array([cv.contourArea(contour) for contour in contours])
    return areas.mean()


def remove_noise(segments, threshold):
    average_area = calc_avg_area(segments)
    remaining_segments = []

    # Remove segments that are below x% of the average area
    for seg in segments:
        if cv.contourArea(seg) < average_area * threshold:
            continue
        remaining_segments.append(seg)

    return remaining_segments


def calc_center(contour):
    M = cv.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    return cx, cy


def rescale_contour(contour, center_x, center_y, scale_size):
    # Rescale the lens contour around its middle point
    # Reduce its area by a factor scale_size
    contour_array = np.array(contour)
    scaled_contour = contour_array - [center_x, center_y]
    scaled_contour = scaled_contour * scale_size
    scaled_contour += [center_x, center_y]
    scaled_contour = (np.rint(scaled_contour)).astype(int)

    return scaled_contour


def resize_image(img, width, height):
    transform_resize = A.Resize(width=width, height=height)

    resized_img = transform_resize(image=img)
    resized_img = resized_img["image"]

    return resized_img


def micron_to_pixel_ratio(scope_mask, og_width, og_height):
    BAR_IN_MICRON = 10

    # original image with ruler for determining the micron to pixel ratio
    annotation_path = "Measurements/Annotations/img83.json"
    image_path = "Measurements/Images/img83.jpg"

    f = open(annotation_path)
    image_segments = json.load(f)

    width_scope = None
    bar_widths = []

    # check specifically annotated segments to get accurate micron to pixel measurements
    for segment in image_segments:
        label = segment["label"]
        area = np.array(segment['points']).astype(int)
        x, y, w, h = cv.boundingRect(area)

        if label == "width_scope":
            width_scope = w
            continue
        if label == "ten_bars":
            bar_widths.append(w/10)
            continue
        if label == "five_bars":
            bar_widths.append(w/5)
            continue
        if label == "one_bar":
            bar_widths.append(w)

    avg_pixels_per_bar = np.mean(bar_widths)
    ruler_micron_pixel_ratio = BAR_IN_MICRON / avg_pixels_per_bar

    # This is the value of the width of the scope of the ruler image in micron
    # This width in micron is identical for all images taken at the same magnification
    scope_in_micron = ruler_micron_pixel_ratio * width_scope

    # Resize scope mask to it's original image size
    # this is mainly done to restore the picture to it's original aspect ratio
    scope_mask = resize_image(scope_mask, og_width, og_height)

    # Find scope contour
    scope_contour, _ = cv.findContours(scope_mask, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_NONE)

    # Calculate the width of the scope in pixels of the scope mask
    _, _, scope_mask_width, _ = cv.boundingRect(scope_contour[0])

    micron_pixel_ratio = scope_in_micron / scope_mask_width

    return micron_pixel_ratio


def split_holes_debris(mask, segments):
    # Filter holes and blocked holes
    segment_dict = {"holes": {}, "debris": {}}

    for i, seg in enumerate(segments):
        # Create a mask for the current segment
        segment_mask = np.zeros_like(mask, dtype=np.uint8)
        cv.drawContours(segment_mask, [seg], -1, 255, cv.FILLED)

        # Calculate the mean value of the filled pixels within the contour
        pixel_value = np.max(mask[segment_mask == 255])

        # Blocked holes by debris
        if pixel_value == 255:
            segment_dict["debris"][f"seg{i}"] = Segment()
            segment_dict["debris"][f"seg{i}"].contour = seg
            continue

        # Open holes
        if pixel_value == 127:
            segment_dict["holes"][f"seg{i}"] = Segment()
            segment_dict["holes"][f"seg{i}"].contour = seg

            # calculate width, height, area of every hole
            area = cv.contourArea(seg)
            (x, y), (w, h), _ = cv.minAreaRect(seg)

            segment_dict['holes'][f"seg{i}"].area = area
            segment_dict['holes'][f"seg{i}"].width = w
            segment_dict['holes'][f"seg{i}"].height = h
            segment_dict['holes'][f"seg{i}"].x = x
            segment_dict['holes'][f"seg{i}"].y = y

    return segment_dict


def cal_avg_area_width_height(segment_dict):
    avg_area = np.mean([seg.area for seg in segment_dict.values()])
    avg_width = np.mean([seg.width for seg in segment_dict.values()])
    avg_height = np.mean([seg.height for seg in segment_dict.values()])

    return avg_area, avg_width, avg_height


def find_outliers(segment_dict, avg_area, avg_width, avg_height, threshold):
    outlier_segments = {}
    max_t = 1 + threshold
    min_t = 1 - threshold

    # Check if area/width/height falls within a margin of the average area
    for key, segment in segment_dict.items():
        if not ((min_t * avg_area) <= segment.area <= (max_t * avg_area)):
            outlier_segments[key] = Segment()
            outlier_segments[key].contour = segment.contour
            outlier_segments[key].x = segment.x
            outlier_segments[key].y = segment.y
            outlier_segments[key].error = "area"
            continue
        if not ((min_t * avg_width) <= segment.width <= (max_t * avg_width)):
            outlier_segments[key] = Segment()
            outlier_segments[key].contour = segment.contour
            outlier_segments[key].x = segment.x
            outlier_segments[key].y = segment.y
            outlier_segments[key].error = "width"
            continue
        if not ((min_t * avg_height) <= segment.height <= (max_t * avg_height)):
            outlier_segments[key] = Segment()
            outlier_segments[key].contour = segment.contour
            outlier_segments[key].x = segment.x
            outlier_segments[key].y = segment.y
            outlier_segments[key].error = "height"

    return outlier_segments


def create_scope_mask(pred_scope):
    # Moprhological closing combines dilation and erosion to improve the shape consistentcy before finding the contour
    # Perform morphological closing to close small gaps and smooth the boundaries of objects.
    kernel_size = 11  # Adjust the kernel size according to your image size
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    scope_closed = cv.morphologyEx(pred_scope, cv.MORPH_CLOSE, kernel)

    # Find the contours in the scope prediction
    contours, _ = cv.findContours(scope_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Find the largest contour -> this being the scope
    scope_contour = max(contours, key=cv.contourArea)

    # Create a binary image with the same size as the original scope image
    scope_mask = np.zeros(pred_scope.shape[:2], dtype=np.uint8)
    # Draw the scope contour as filled white pixels on the binary image
    cv.drawContours(scope_mask, [scope_contour], 0, (255), thickness=cv.FILLED)

    return scope_mask, scope_contour


def scale_scope(pred_scope, scope_contour, scale_size):
    # Calculate the center point of scope contour
    scope_center_x, scope_center_y = calc_center(scope_contour)

    # Scale the contour of the scope around it's center point
    # Reduce its area by 20% (default)
    scaled_scope_contour = rescale_contour(scope_contour, scope_center_x, scope_center_y, scale_size)

    # Create a scaled scope mask
    scaled_scope_mask = np.zeros(pred_scope.shape[:2], dtype=np.uint8)
    # Draw the scaled scope contour as filled white pixels on the binary image
    cv.drawContours(scaled_scope_mask, [scaled_scope_contour], 0, (255), thickness=cv.FILLED)

    return scaled_scope_mask


def post_process(pred_sieve, pred_scope, true_label=None, eval=False):
    #Input the a predicted hole segment mask, scope mask to perform post processing
    #Optionally also post_process the corresponding true label mask for evalutation
    scope_mask, scope_contour = create_scope_mask(pred_scope)

    # Find all segments in the predicted mask image
    mask_contours, _ = cv.findContours(pred_sieve, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Remove all segments from the mask prediction image that fall outside the scope area
    # Thus remove all background segments
    remaining_mask_segments = remove_background(mask_contours, scope_mask)

    # Create a scaled scope mask reducing its area by 20%
    scaled_scope_mask = scale_scope(pred_scope, scope_contour, scale_size=0.8)

    # Remove all (remaining) sieve segments that fall partially or completely outside the scaled scope
    # with a margin of 5%
    segments_of_interest = remove_segments(remaining_mask_segments, scaled_scope_mask, 0.05)

    # Remove segments that have an area of less than 50% the average area
    remaining_segments = remove_noise(segments_of_interest, 0.5)

    # Create the segments of interest as a new image
    # Create a blank image with the same size as the mask image
    new_mask = np.zeros_like(pred_sieve)
    # Fill the mask with only the contour area's
    for seg in remaining_segments:
        cv.fillPoly(new_mask, [seg], color=(255, 255, 255))
    # Perform a bitwise and operation between the original image and  the mask
    # Restoring the original mask pixel values ast the positoin of the segemnts
    # Keeping the rest of the mask blank
    new_mask = cv.bitwise_and(pred_sieve, new_mask)


    # Fuse segments where 25% of the total pixels are off the debris class together
    # Making the entire segment debris, else make the segment hole
    new_mask = fuse_segments(remaining_segments, pred_sieve, new_mask, 0.25)

    # Not in eval mode only return the post processed sieve mask
    if not eval:
        return new_mask, scope_mask, scaled_scope_mask

    # Remove the edge holes
    # Find all segments in the true mask image
    true_contours, _ = cv.findContours(true_label, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Remove all (remaining) sieve segments that fall partially or completely outside the scaled scope
    # with a margin of 5%
    true_seg_interest = remove_segments(true_contours, scaled_scope_mask, 0.05)

    # Create ground truth mask for region of interest
    new_true = np.zeros_like(true_label)
    for seg in true_seg_interest:
        cv.fillPoly(new_true, [seg], color=(255, 255, 255))
    new_true = cv.bitwise_and(true_label, new_true)

    return new_mask, new_true


def write_csv_report(csv_path, input1=None, input2=None, input3=None, input4=None,create_header1=False, create_header2=False):
    # Create new header for train / validation set
    if create_header1:
        header = ["debris_quantity", "mean_area", "mean_width", "mean_height"]
        with open(csv_path, "w", encoding="UTF8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerow([input1, input2, input3, input4])

        return

    if create_header2:
        header = ["segment_center", "area", "width", "height"]
        with open(csv_path, "a", encoding="UTF8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
        return

    # Add data row to train / validation csv
    with open(csv_path, "a", encoding="UTF8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([input1, input2, input3, input4])
