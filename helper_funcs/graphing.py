# required imports
import cv2 as cv


# def normalise(x, y, width, height):
#     return [int(x * width), int(y * height)]


class GraphHelper:
    """
        Class to do graphing functions
    """

    def draw_pose(self, frame, points):
        """
        Will draw unnormalised coordinates on frame
        :param points:
        :param frame:
        """
        temp = frame.copy()
        for point in points:
            # cv.drawMarker(temp, point, color=(0, 255, 0), markerType=cv.MARKER_DIAMOND, thickness=5)
            cv.circle(temp, point, 3, (0, 0, 255), -1)

        cv.imshow("Points", temp)
