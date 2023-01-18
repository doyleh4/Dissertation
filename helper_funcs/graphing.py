# required imports
import cv2 as cv
import matplotlib.pyplot as plt


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
        Will draw the points as we want to check them for front view
        :param frame:
        :param points:
        """

        """
            Indexes - 0: left wrist, 1: right wrist, 2: left elbow, 3: right elbow, 4: left shoulder, 5: right shoulder,
            6: left hip, 7: right hip, 8: left knee, 9: right knee, ,10: left ankle, 11: right ankle, 12: right heel,
            13: right foot index/toe, 14: left ear, 15: nose
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

        # Right foot
        cv.line(temp, points[12], points[13], (0, 0, 255), 5)

        # Lead foot inline to head
        cv.line(temp, points[10], points[6], (255, 255, 0), 4)  # left foot to left hip
        cv.line(temp, points[6], points[4], (255, 255, 0), 4)  # left hip to lef shoulder
        cv.line(temp, points[4], points[15], (255, 255, 0), 4)  # left shoulder to nose

        # Lead foot, hip to shoulder inline (similar to above but different check)
        cv.line(temp, points[10], points[6], (255, 0, 0), 2)  # left foot to left hip
        cv.line(temp, points[6], points[4], (255, 0, 0), 2)  # left hip to lef shoulder

        cv.imshow("Expanded checks", temp)

    def draw_dtl_checks(self, frame, points):
        """
        Will draw the points as we want to check them for down the line view
        :param frame:
        :param points:
        """
        """
            Indexes - 0: right hip, 1: right knee, 2: right ankle, 3: right shoulder, 4: right elbow, 5: right wrist,
            6: left shoulder, 7: left hip, 8: right heel, 9: right foot index/toe
        """
        temp = frame.copy()

        # Right knee slightly bent
        cv.line(temp, points[0], points[1], (0, 255, 0), 2)  # right hip to knee
        cv.line(temp, points[1], points[2], (0, 255, 0), 2)  # right knee to ankle
        cv.circle(temp, points[1], 3, (0, 0, 255), -1)  # right knee

        # Right arm direction
        cv.line(temp, points[5], points[4], (255, 255, 0), 5)  # right wrist to elbow

        # Right elbow slightly bent
        cv.line(temp, points[3], points[4], (255, 0, 0), 2)  # right shoulder to elbow
        cv.line(temp, points[4], points[5], (255, 0, 0), 2)  # right elbow to wrist
        cv.circle(temp, points[1], 3, (0, 0, 255), -1)  # right elbow

        # Shoulder slope
        cv.line(temp, points[3], points[4], (0, 255, 255), 2)  # left to right shoulder

        # Hip slope
        cv.line(temp, points[0], points[7], (255, 0, 0), 2)  # right shoulder to elbow

        # Right foot
        cv.line(temp, points[8], points[9], (255, 0, 255), 2)  # right shoulder to elbow

        cv.imshow("Expanded checks", temp)

    def show_graphs(self, data):
        fig, ax = plt.subplots()

        tempX = []
        tempY = []
        for item in data.data['lw']:
            # Open CV uses X,Y - where X is column, Y is row
            tempX.append(item[0])
            tempY.append(item[1])
        plt.margins(1, 2.8)  # set margins to approximately be the same as opencv window
        plt.plot(tempX, tempY)
        plt.show()

        tempX = data.data["shoulder_slope"]
        plt.plot(tempX)
        plt.show()

        tempX = data.data["lead_leg_to_shoulder"]
        plt.plot(tempX)
        plt.show()
