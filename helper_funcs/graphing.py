# required imports
import cv2 as cv


# def normalise(x, y, width, height):
#     return [int(x * width), int(y * height)]


class GraphHelper:
    """
        Class to do graphing functions
    """

    def draw_pose_results(self, frame, points):
        """
        Will draw landmark coordinates on frame
        :param frame:
        :param points:
        """
        temp = frame.copy()
        for point in points:
            cv.circle(temp, point, 3, (0, 0, 255), -1)

        cv.imshow("Points", temp)

    def draw_expanded(self, frame, points):
        """
        Will draw the points as we want to check them
        :param frame:
        :param points:
        """
        temp = frame.copy()

        # Shoulders
        cv.line(temp, points[4], points[5], (0, 0, 255), 10)  # left shoulder to right shoulder

        # Simultaneous rotation - one piece
        cv.line(temp, points[4], points[5], (0, 255, 0), 2)  # again for different check
        cv.line(temp, points[4], points[2], (0, 255, 0), 2)  # left shoulder to left elbow
        cv.line(temp, points[2], points[0], (0, 255, 0), 2)  # left elbow to left wrist
        cv.line(temp, points[5], points[3], (0, 255, 0), 2)  # right shoulder to right elbow
        cv.line(temp, points[3], points[1], (0, 255, 0), 2)  # right elbow to right wrist

        # Degree of rotation at elbows
        cv.circle(temp, points[2], 3, (0, 0, 255), -1)  # left elbow
        cv.circle(temp, points[3], 3, (0, 0, 255), -1)  # right elbow

        # Lead foot, hip to shoulder inline
        cv.line(temp, points[10], points[6], (255, 0, 0), 2)  # left foot to left hip
        cv.line(temp, points[6], points[4], (255, 0, 0), 2)  # left hip to lef shoulder

        # Right foot
        # Todo - right foot tracking is very bad. Try fix it
        cv.line(temp, points[12], points[13], (0, 0, 255), 5)

        cv.imshow("Connected attributes", temp)
