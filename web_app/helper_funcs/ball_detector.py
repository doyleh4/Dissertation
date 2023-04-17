import cv2 as cv
import numpy as np


def detect(frame):
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    blurred = cv.GaussianBlur(gray, (3, 3), 0)

    ret, mask = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    ellipses = []
    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv.fitEllipse(cnt)
            center, axes, angle = ellipse
            major_axis, minor_axis = axes

            # Filter out larger ellipses
            if 5 <= major_axis <= 9 and 5 <= minor_axis <= 9:
                # cv.ellipse(frame, ellipse, (0, 0, 255), 2)
                ellipses.append(ellipse)

    # Find the most circular ellipse
    try:
        aspect_ratios = []
        for ellipse in ellipses:
            length = ellipse[1][0]
            height = ellipse[1][1]
            aspect_ratio = length / height
            aspect_ratios.append(aspect_ratio)

        index = aspect_ratios.index(min(aspect_ratios, key=lambda x: abs(x - 1)))
        most_circular = ellipses[index]

        # cv.ellipse(frame, most_circular_ellipse, (255, 0, 0), 3)
        return most_circular
    except:
        return None


class BallDetector:
    pass
