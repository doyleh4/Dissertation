# required imports
import math

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# from helper_funcs.classify_stage import calculate_acceleration


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
    # arr = coords[indices]
    # arr[:, 1] = -arr[:, 1]

    # Keep only indices (makes array shorter)
    # res = coords[indices]
    # res = np.vstack([res, [None, None]])

    # Change index not in indices to None
    res = np.empty_like(coords, dtype=object)
    res[indices, :] = coords[indices, :]

    return res

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


def estimate_missing_points(points):
    """
    If an array with missing rows is passed in after outliers are removed, this function will estimate
    the "realistic" values of the outlier points. For now these estimated values are lines, instead of curves.
    While this will decrease the accuarcy, it is minimal.
    :param filtered:
    :return: estimated array
    """
    # Using mean imputation for missing rows.
    # mean_imputer = SimpleImputer(strategy="mean")
    # data_imputed = mean_imputer.fit_transform(filtered)

    # Using variance thresholding
    # var_thresh = VarianceThreshold(threshold=0.1)
    # data_imputed = var_thresh.fit_transform(filtered)

    # # Use KNN to estimate the data from the mean
    # knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
    # knn_imputer.fit(filtered)
    #
    # res = knn_imputer.transform(filtered)

    # Iterate over the points
    for i in range(1, len(points) - 1):
        # If the current point is missing
        if points[i][0] is None and points[i][1] is None:
            # Find the closest known points before and after
            j = i - 1
            while j >= 0 and points[j][0] is None and points[j][1] is None:
                j -= 1
            k = i + 1
            while k < len(points) and points[k][0] is None and points[k][1] is None:
                k += 1
            # Calculate the slope and intercept between the known points
            # x1, y1 = points[j]
            # x2, y2 = points[k]
            # m = (y2 - y1) / (x2 - x1)
            # b = y1 - m * x1
            # # Estimate the missing point using the linear equation
            # # x = points[i - 1, 0]
            # y = m * (i + 1) + b
            # points[i] = [i + 1, y]

            if (k < len(points)):
                start_point = points[j]
                end_point = points[k]

                n = k - j + 1

                dx = (end_point[0] - start_point[0]) / (n - 1)
                dy = (end_point[1] - start_point[1]) / (n - 1)

                t = np.array([[start_point[0] + i * dx, start_point[1] + i * dy] for i in range(n)])

                points[j:k + 1, :] = t

    return points


def calculate_acceleration(filled):
    acc = []
    if len(filled) > 1:
        for index in range(1, len(filled)):
            p1 = filled[index]
            p2 = filled[index - 1]
            dist = math.dist(p1, p2)
            acc.append(dist)

    return acc


class GraphHelper:
    """
        Class to do graphing functions
    """

    def __init__(self):
        """
        These attributes will store the processed data.
        """
        self.proc_fo_data = None
        self.proc_dtl_data = None

    def set_processed_fo_data(self, data):
        self.proc_fo_data = data

    def get_processed_fo_data(self):
        return self.proc_fo_data

    def set_processed_dtl_data(self, data):
        self.proc_dtl_data = data

    def get_processed_dtl_data(self):
        return self.proc_dtl_data

    def draw_pose_results(self, frame, points):
        """
        Will draw landmark coordinates on frame
        :param frame:
        :param points:
        """
        temp = frame.copy()
        for point in points:
            cv.circle(temp, point, 3, (0, 0, 255), -1)

        # cv.imshow("Points", temp)

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

        # cv.imshow("Expanded checks", temp)
        return temp

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
        # cv.ellipse(temp, ball, (0, 0, 255), 2)

        # cv.imshow("Expanded checks", temp)
        return temp

    def show_graphs(self, data, dtl_data, t):
        # fig, ax = plt.subplots()
        # ax.invert_yaxis()  # Inverting y axis to allow the coordinate system match OpenCV
        # #
        # tempX = [val[0] for val in dtl_data.r_wrist]
        # tempY = [val[1] for val in dtl_data.r_wrist]
        # # TODO: t - val[1] seems to shove it below the y axis so fix this
        # plt.margins(1, 2.8)  # set margins to approximately be the same as opencv window
        # curve, = plt.plot(tempX, tempY)
        # plt.scatter(tempX[0], tempY[0])
        # plt.show()
        #
        filtered = remove_outliers(data.data['lw'])  # Delete outlier points
        filtered2 = remove_outliers(dtl_data.r_wrist)
        # tempX = [val[0] for val in filtered2]
        # tempY = [val[1] for val in filtered2]
        # # TODO: t - val[1] seems to shove it below the y axis so fix this
        # plt.margins(1, 2.8)  # set margins to approximately be the same as opencv window
        # curve, = plt.plot(tempX, tempY)
        # plt.scatter(tempX[0], tempY[0])

        # Fit a curve to those points
        # coeffs = np.polyfit(tempX, tempY, 29)
        # x_fit = np.linspace(min(tempX), max(tempX), 100)
        # y_fit = np.polyval(coeffs, x_fit)
        # plt.plot(x_fit, y_fit, "r")
        # plt.show()

        # # TODO: Move this above, just want both graphs for cdev
        filled = estimate_missing_points(filtered)
        filled2 = estimate_missing_points(filtered2)
        # fig, ax = plt.subplots()
        # tempX = [val[0] for val in filled2]
        # tempY = [val[1] for val in filled2]
        # ax.invert_yaxis()  # Inverting y axis to allow the coordinate system match OpenCV

        # Calulate these accelerations
        acceleration = calculate_acceleration(filled)
        acceleration2 = calculate_acceleration(filled2)

        # Display these acceleration graphs
        # y = np.arange(len(acceleration))
        # plt.plot(y, acceleration, label="Face on")
        #
        # y = np.arange(len(acceleration2))
        # plt.plot(y, acceleration2, label="Down the line")
        # plt.legend()
        # plt.show()

        # tempX = [val[0] for val in filtered2]
        # tempY = [val[1] for val in filtered2]
        # # TODO: t - val[1] seems to shove it below the y axis so fix this
        # plt.margins(1, 2.8)  # set margins to approximately be the same as opencv window
        # curve, = plt.plot(tempX, tempY)
        # plt.scatter(tempX[0], tempY[0])

        # tempX = [val[0] for val in filtered2]
        # tempY = [val[1] for val in filtered2]
        # # TODO: t - val[1] seems to shove it below the y axis so fix this
        # plt.margins(1, 2.8)  # set margins to approximately be the same as opencv window
        # curve, = plt.plot(tempX, tempY)
        # plt.scatter(tempX[0], tempY[0])

        plt.show()

        self.set_processed_fo_data(acceleration)
        self.set_processed_dtl_data(acceleration2)
        # #
        # plt.margins(1, 2.8)  # set margins to approximately be the same as opencv window
        # curve, = plt.plot(tempX, tempY)
        # plt.scatter(tempX[0], tempY[0])
        # y_vals = curve.get_ydata()
        # plt.show()
        #
        # tempX = data.data["shoulder_slope"]
        # plt.plot(tempX)
        # plt.show()
        #
        # tempX = data.data["lead_leg_to_shoulder"]
        # plt.plot(tempX)
        # plt.show()
        #
        # tempX = data.data["acc"]
        # plt.plot(tempX)
        # plt.show()

    def get_acceleration(self, data):
        """
        Function to calculate acceleration of the wrist
        :param data:
        :return:
        """
        try:
            filtered = remove_outliers(data.data['lw'])  # Delete outlier points
        except:
            filtered = remove_outliers(data.r_wrist)
        # Yes these are 2 different data types. Its horrible I know, im tired leave me alone :(

        filled = estimate_missing_points(filtered)

        # Note: this is only needed in website version of app. Runs ok when normal
        filled = np.delete(filled, -1, axis=0)
        # Calulate these accelerations
        acceleration = calculate_acceleration(filled)

        return acceleration

    # Visualise the checks
    def leg_width_check(self, frame, ankles, shoulders):
        """
        A function to display the leg checks
        :param frame:
        :param shoulders:
        :param ankles:
        :return:
        """
        temp = frame.copy()

        # Mark Points
        cv.circle(temp, ankles[0], 3, (0, 0, 255), -1)
        cv.circle(temp, ankles[1], 3, (0, 0, 255), -1)
        cv.circle(temp, shoulders[0], 3, (0, 0, 255), -1)
        cv.circle(temp, shoulders[1], 3, (0, 0, 255), -1)

        # Draw line straight down from the shoulders
        cv.line(temp, ankles[0], shoulders[0], (255, 255, 0), 1)
        cv.line(temp, ankles[1], shoulders[1], (255, 255, 0), 1)

        # cv.imshow("Leg check", temp)
        cv.imwrite("video/leg_width.jpg", temp)

    def one_piece_movement_check(self, frame, wrists, elbows, shoulders):
        """
        Display the arms and shoulders moving together
        :param frame:
        :param wrists:
        :param shoulders:
        :return:
        """
        temp = frame.copy()

        cv.line(temp, shoulders[0], shoulders[1], (0, 255, 0), 2)  # shoulder to shoulder (COYBIG)
        cv.line(temp, shoulders[0], elbows[0], (0, 255, 0), 2)  # shoulder to wrist
        cv.line(temp, shoulders[1], elbows[1], (0, 255, 0), 2)
        cv.line(temp, elbows[0], wrists[0], (0, 255, 0), 2)  # elbow to wrist
        cv.line(temp, elbows[1], wrists[1], (0, 255, 0), 2)

        # cv.imshow("OPM check", temp)
        cv.imwrite("video/one_piece_movement.jpg", temp)

    def shoulder_over_foot(self, frame, shoulder, foot):
        """
        Dispaly the shoulder foot location in the followthrough
        :param frame:
        :param shoulder:
        :param foot:
        :return:
        """
        temp = frame.copy()

        cv.line(temp, shoulder, foot, (255, 255, 0), 1)

        # cv.imshow("Shoulder over foot check", temp)
        cv.imwrite("video/shoulder_over_foot.jpg", temp)

    def head_behind_ball(self, frame, ball, head):
        """
        Function to display head behind ball check
        :param frame:
        :param ball:
        :param head:
        :return:
        """
        temp = frame.copy()

        # cv.ellipse(temp, ball, (0, 0, 255), 2)
        cv.circle(temp, [int(ball[0]), int(ball[1])], 6, (0, 0, 255), -1)
        cv.circle(temp, head, 6, (0, 0, 255), -1)

        cv.line(temp, head, [int(ball[0]), int(ball[1])], (255, 255, 0), 1)

        # cv.imshow("Head behind ball check", temp)
        cv.imwrite("video/head_behind_ball.jpg", temp)

    def knee_angle(self, frame, ankle, knee, hip, stage):
        """
        Function to display head behind ball check
        :param frame:
        :param ankle:
        :param knee:
        :param hip:
        :return:
        """
        temp = frame.copy()

        cv.circle(temp, ankle, 6, (0, 0, 255), -1)
        cv.circle(temp, knee, 6, (0, 0, 255), -1)
        cv.circle(temp, hip, 6, (0, 0, 255), -1)

        cv.line(temp, ankle, knee, (255, 0, 0), 3)
        cv.line(temp, knee, hip, (255, 0, 0), 3)

        # cv.imshow("Knee angle check", temp)
        cv.imwrite("video/{}_knee_angle.jpg".format(stage), temp)

    def trail_arm_straight(self, frame, wrist, elbow, shoulder):
        temp = frame.copy()

        cv.circle(temp, wrist, 6, (0, 0, 255), -1)
        cv.circle(temp, elbow, 6, (0, 0, 255), -1)
        cv.circle(temp, shoulder, 6, (0, 0, 255), -1)

        cv.line(temp, wrist, elbow, (255, 0, 0), 3)
        cv.line(temp, elbow, shoulder, (255, 0, 0), 3)

        # cv.imshow("Train arm straight check", temp)
        cv.imwrite("video/trail_arm_straight.jpg", temp)

    def shoulder_slope(self, frame, shoulders, arm):
        temp = frame.copy()

        cv.circle(temp, shoulders[0], 6, (0, 0, 255), -1)
        cv.circle(temp, shoulders[1], 6, (0, 0, 255), -1)

        cv.line(temp, shoulders[0], shoulders[1], (255, 0, 0), 3)

        cv.circle(temp, arm[0], 6, (0, 0, 255), -1)
        cv.circle(temp, arm[1], 6, (0, 0, 255), -1)

        cv.line(temp, arm[0], arm[1], (255, 0, 0), 3)

        # cv.imshow("Left arm in same plane as shoulders", temp)
        cv.imwrite("video/left_arm_plane.jpg", temp)

    def elbow_pointing_down(self, frame, elbow, shoulder):
        temp = frame.copy()

        # TODO: Again this is a check to see the left arm plane (inline with shoulder slope), not just shoulder slope
        cv.circle(temp, shoulder, 6, (0, 0, 255), -1)
        cv.circle(temp, elbow, 6, (0, 0, 255), -1)

        cv.arrowedLine(temp, shoulder, elbow, (255, 0, 0), 3)

        # cv.imshow("Trail elbow pointing down", temp)
        cv.imwrite("video/elbow_pointing_down.jpg", temp)

    def shoulders_closed(self, frame, shoulders):
        # Very similar to shoulder slope above so maybe change
        temp = frame.copy()

        # TODO: Again this is a check to see the left arm plane (inline with shoulder slope), not just shoulder slope
        cv.circle(temp, shoulders[0], 6, (0, 0, 255), -1)
        cv.circle(temp, shoulders[1], 6, (0, 0, 255), -1)

        cv.line(temp, shoulders[0], shoulders[1], (255, 0, 0), 3)

        # cv.imshow("Shoulders slightly closed", temp)
        cv.imwrite("video/shoulders_closed.jpg", temp)
