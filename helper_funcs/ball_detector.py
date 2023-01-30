import cv2 as cv


def detect(frame):
    # Hough circle thing
    cv.imshow("Temp", frame)

    # Read image as gray-scale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv.threshold(blur, 210, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    cv.imshow("Thresh", thresh)

    # img_blur = cv.GaussianBlur(gray, (15, 15), 0)
    # #
    # cv.imshow("Temp", img_blur)
    #
    # # Apply hough transform on the gray scale image
    # # Change the parameters to match the circle that is needed
    # circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 2, 1, param1=100, param2=30, minRadius=1,
    #                           maxRadius=10)
    #
    # # Draw detected circles
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         # Change the parameters to select the circle that is needed in the frame
    #         # Here I am selecting only the circles above 300 px in the x axis
    #         if i[0] > 300:
    #             # print('Circle center - {} , {} -- Radius {}'.format(i[0],i[1],i[2]))
    #             # Draw circle around the ball
    #             cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #
    # cv.imshow("Temp ball", frame)

    return frame


class BallDetector:
    pass
