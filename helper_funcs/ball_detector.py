import cv2 as cv
import numpy as np

template = cv.imread("template.JPG")


def detect(frame):
    # Template matching works but isnt the best because of scale

    # temp = frame.copy()
    #
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # t_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    #
    # res = cv.matchTemplate(gray, t_gray, cv.TM_CCOEFF_NORMED)
    #
    # _, max_val, _, max_loc = cv.minMaxLoc(res)
    #
    # h, w, _ = template.shape
    # top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv.rectangle(temp, top_left, bottom_right, (0, 0, 255), 2)
    #
    # cv.imshow("temp", temp)

    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    blurred = cv.GaussianBlur(gray, (3, 3), 0)

    ret, mask = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # cv.imshow("temp", mask)

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
            major_axis_length = ellipse[1][0]
            minor_axis_length = ellipse[1][1]
            aspect_ratio = major_axis_length / minor_axis_length
            aspect_ratios.append(aspect_ratio)

        index_of_most_circular_ellipse = aspect_ratios.index(min(aspect_ratios, key=lambda x: abs(x - 1)))
        most_circular_ellipse = ellipses[index_of_most_circular_ellipse]

        # cv.ellipse(frame, most_circular_ellipse, (255, 0, 0), 3)
        return most_circular_ellipse
    except:
        return None


class BallDetector:
    pass
