# required imports
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# def normalise(x, y, width, height):
#     return [int(x * width), int(y * height)]
def smooth_line(points, window_size=5):
    line = np.array(points)
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(line, window, 'same')


def remove_outliers(data):
    """
    A function to remove outlier points from the data
    Uses a z-score and threshold approach to get rid of outliers

    Note: This doesnt work perfectly, it will delete some of the handpoints in the followthrough
    but because we are not verifying the location of hand in that stage this is ok but we may need to
    change for other tracking points. Hopefully not

    TODO: Make this work for all tracked points
    """
    thresh = 0.95
    coords = np.array(data)

    # Calculate z-scores and get outliers
    z_scores = np.abs((coords - np.mean(coords, axis=0)) / np.std(coords, axis=0))
    outliers = np.argwhere(z_scores > thresh)
    indices = np.unique(outliers[:, 0])

    # We need to return the y values as negative, to offset the differnce of MATPLOTLIB and opencv
    arr = coords[indices]
    arr[:, 1] = -arr[:, 1]
    return coords[indices]

    # # DEBUG Stuff # TODO delete this
    # t = np.arange(coords.shape[0])
    # missing = np.setxor1d(t, indices)
    #
    # print("Z scores at indexes are")
    # for i, index in enumerate(missing):
    #     print("{}:{}".format(index, z_scores[index]))
    #
    # temp_data = np.delete(coords, indices, axis=0)
    # x = [val[0] for val in temp_data]
    # y = [val[1] for val in temp_data]
    # plt.scatter(x, y)
    # plt.margins(1, 2.8)
    # plt.show()


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

    def draw_expanded(self, frame, points, ball):
        """
        Will draw the points as we want to check them for front view
        :param frame:
        :param points:
        :param ball:
        """

        """
            Indexes - 0: left wrist, 1: right wrist, 2: left elbow, 3: right elbow, 4: left shoulder, 5: right shoulder,
            6: left hip, 7: right hip, 8: left knee, 9: right knee, ,10: left ankle, 11: right ankle, 12: right heel,
            13: right foot index/toe, 14: left ear, 15: nose
        """
        # TODO: Change nose here to left ear
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
        cv.line(temp, points[10], points[8], (255, 255, 0), 4)  # left foot to left knee
        cv.line(temp, points[8], points[6], (255, 255, 0), 4)  # left knee to left hip
        cv.line(temp, points[6], points[4], (255, 255, 0), 4)  # left hip to lef shoulder
        cv.line(temp, points[4], points[15], (255, 255, 0), 4)  # left shoulder to nose
        # Draw the individual points that are being tracked after next check top avoid over lapping

        # Lead foot, hip to shoulder inline (similar to above but different check)
        cv.line(temp, points[10], points[6], (255, 0, 0), 2)  # left foot to left hip
        cv.line(temp, points[6], points[4], (255, 0, 0), 2)  # left hip to lef shoulder

        cv.circle(temp, points[10], 3, (255, 0, 255), -1)
        cv.circle(temp, points[8], 3, (255, 0, 255), -1)
        cv.circle(temp, points[6], 3, (255, 0, 255), -1)
        cv.circle(temp, points[4], 3, (255, 0, 255), -1)

        cv.circle(temp, points[0], 5, (0, 255, 255), -1)  # left wrist

        # Draw the balls location (for now)
        # TODO: Update this to detect in the first frame and track it after that
        cv.ellipse(temp, ball, (0, 0, 255), 2)

        cv.imshow("Expanded checks", temp)

    def draw_dtl_checks(self, frame, points, ball):
        """
        Will draw the points as we want to check them for down the line view
        :param frame:
        :param points:
        :param ball:
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

        # Draw the balls location (for now)
        # TODO: Update this to detect in the first frame and track it after that
        cv.ellipse(temp, ball, (0, 0, 255), 2)

        cv.imshow("Expanded checks", temp)

    def show_graphs(self, data, t):
        fig, ax = plt.subplots()

        filtered = remove_outliers(data.data['lw'])  # Delete outlier points
        tempX = [val[0] for val in filtered]
        tempY = [val[1] for val in
                 filtered]  # t is to match the coordinate system of opencv and matplotlib
        # TODO: t - val[1] seems to shove it below the y axis so fix this
        plt.margins(1, 2.8)  # set margins to approximately be the same as opencv window
        curve, = plt.plot(tempX, tempY)
        plt.scatter(tempX[0], tempY[0])

        # Fit a curve to those points
        coeffs = np.polyfit(tempX, tempY, 29)
        x_fit = np.linspace(min(tempX), max(tempX), 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.invert_yaxis()
        y_vals = curve.get_ydata()
        plt.plot(x_fit, y_fit, "r")
        plt.show()

        tempX = data.data["shoulder_slope"]
        plt.plot(tempX)
        plt.show()

        tempX = data.data["lead_leg_to_shoulder"]
        plt.plot(tempX)
        plt.show()

        tempX = data.data["acc"]
        plt.plot(tempX)
        plt.show()
