# # Install packages
# # OS
# import argparse
# import sys
#
# # Essenital
# import cv2 as cv
# import mediapipe as mp
# import numpy as np
# from mediapipe.framework.formats import landmark_pb2
# # Helpers
# from scipy.interpolate import splprep, splev
#
# # Retrieve input file from run-time ram
# parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
#                                               OpenCV. You can process both videos and images.')
# parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.',
#                     default="./videos/samples/sample0.mov")
# args = parser.parse_args()
#
# print("Retrieiving video from files")
#
# capture = cv.VideoCapture(args.input)
# compare_capture = cv.VideoCapture("./videos/tiger.mp4")
#
# if not capture.isOpened():
#     print("Input file failed to open, there is a file path error.")
#     sys.exit(0)
#
# if not compare_capture.isOpened():
#     print("Comparison file failed to open, there is a file path error.")
#     sys.exit(0)
#
# subtraction = cv.createBackgroundSubtractorMOG2()
#
# # TODO: Classify all similar functions
#
# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils
#
# # initialize the HOG descriptor/person detector
# hog = cv.HOGDescriptor()
# hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
#
#
# def detect_person(frame):
#     """
#     method to detect a person via contours https://learnopencv.com/contour-detection-using-opencv-python-c/#:~:text=to%20add%20content.-,Contour%20Detection%20using%20OpenCV%20(Python%2FC%2B%2B),image%20segmentation%2C%20detection%20and%20recognition.
#     :param frame:
#     :return: result
#     """
#     res = frame.copy()
#     # res = cv.resize(res, (640, 480))
#     gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
#     # Ignore the sky
#     # gray[gray > 220] = 0
#     cv.imshow("Gray", gray)
#
#     ret, thresh = cv.threshold(gray, 185, 255, cv.THRESH_BINARY)
#     # ret, thresh = cv.threshold(gray, 65, 255, cv.THRESH_BINARY_INV)
#     thresh = cv.medianBlur(thresh, 9)
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
#     thresh = cv.dilate(thresh, kernel, iterations=1)
#
#     cv.imshow("bin", thresh)
#
#     # Centre of mass computation via numpy
#     thresh = thresh / 255
#     thresh = thresh.astype(np.uint8)
#     # print(np.amax(thresh), np.amin(thresh), thresh.dtype)
#     # cv.imshow("Temp", thresh * 255)
#
#     coords = np.where(thresh == 1)
#     coords = np.array([coords[0], coords[1]])
#     coords = coords.T
#
#     xs_average = int(np.sum(coords[:, 0]) / len(coords))
#     ys_average = int(np.sum(coords[:, 1]) / len(coords))
#
#     # print(xs_average)
#     # print(ys_average)
#     cv.circle(res, (xs_average, ys_average), 1, (0, 255, 255), 5)
#     cv.imshow("COM", res)
#
#     # Centre of mass computation via OpenCV
#     contours, hieracrchy = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
#     cv.drawContours(gray, contours, -1, (0, 255, 0), 5, cv.LINE_AA)
#
#     # cv.imshow("Contours", gray)
#
#     largest = []
#     for contour in contours:
#         if len(contour) > len(largest):
#             largest = contour
#     image_copy = frame.copy()
#     cv.drawContours(image=image_copy, contours=largest, contourIdx=-1, color=(0, 255, 0), thickness=2,
#                     lineType=cv.LINE_AA)
#     return largest
#
#
# def centre_of_mass(frame):
#     """
#     Function to detect the centre of mass of the player
#     Reference - https://stackoverflow.com/questions/49582008/center-of-mass-in-contour-python-opencv
#     :param frame:
#     :return: centre of mass as a circle
#     """
#     com = {
#         "c_x": 0,
#         "c_y": 0
#     }
#     res = frame.copy()
#     person_cont = detect_person(res)
#
#     res_copy = res.copy()
#     cv.drawContours(image=res_copy, contours=person_cont, contourIdx=-1, color=(0, 255, 0), thickness=2,
#                     lineType=cv.LINE_AA)
#     # cv.imshow("Player", res_copy)
#
#     M = cv.moments(person_cont)
#     com["c_x"] = int(M["m10"] / M["m00"])
#     com["c_y"] = int(M["m01"] / M["m00"])
#
#     # Retruning it seems to be offset but not sure. Maybe this mehtod doesnt woirk - implemented better in detect_person()
#
#     return com
#
#
#
# def hand_estimation(frame):
#     # inspired by https://www.analyticsvidhya.com/blog/2021/05/pose-estimation-using-opencv/
#     res = frame.copy()
#     results = pose.process(frame)
#
#     # Get the subset of landmarks that I need for the problem i.e. just the arms
#     landmark_subset = landmark_pb2.NormalizedLandmarkList(
#         landmark=[
#             results.pose_landmarks.landmark[11],
#             results.pose_landmarks.landmark[12],
#             results.pose_landmarks.landmark[13],
#             results.pose_landmarks.landmark[14],
#             results.pose_landmarks.landmark[15],
#             results.pose_landmarks.landmark[13],
#             results.pose_landmarks.landmark[17],
#             results.pose_landmarks.landmark[18],
#             results.pose_landmarks.landmark[19],
#             results.pose_landmarks.landmark[20],
#
#         ]
#     )
#
#     # draw landmarks
#     mpDraw.draw_landmarks(
#         image=res,
#         landmark_list=landmark_subset)
#     return res
#
#
# def find_corners(grayframe):
#     """
#     Function to find the corners in the frame. Influenced by:
#     https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html
#     :param grayframe:
#     :return:
#     """
#     corners = cv.goodFeaturesToTrack(grayframe, 100, 0.1, 15)
#     corners = np.int0(corners)
#
#     return corners
#
#
# def kmeans_clustering(frame):
#     Z = frame.reshape((-1, 3))
#     # convert to np.float32
#     Z = np.float32(Z)
#     # define criteria, number of clusters(K) and apply kmeans()
#     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     K = 5
#     ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
#
#     # Now convert back into uint8, and make original image
#     center = np.uint8(center)
#     res = center[label.flatten()]
#     res2 = res.reshape(frame.shape)
#     # cv.imshow('res2', res2)
#     res2 = cv.cvtColor(res2, cv.COLOR_BGR2GRAY)
#     return res2
#
#
# def bin_threshold(frame):
#     ret, thresh = cv.threshold(frame, 90, 255, cv.THRESH_BINARY_INV)
#
#     kernel = np.ones((2, 2), np.uint8)
#     thresh = cv.erode(thresh, kernel, iterations=1)
#     kernel = np.ones((2, 2), np.uint8)
#     thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
#     kernel = np.ones((5, 5), np.float32) / 25
#     thresh = cv.filter2D(thresh, -1, kernel)
#
#     # g_frame = frame.copy()
#     # result = cv.bitwise_and(g_frame, thresh)
#
#     contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
#     temp = frame.copy()
#     # for index in range(len(contours)):
#     #     contour = contours[index]
#     #     approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
#     #     # if len(approx) is 3:
#     #     #     print("hit")
#     #     if 10 < len(approx) < 25:
#     #         temp = cv.drawContours(thresh, [approx], 0, 65, 3)
#     #     # print(temp)
#     #     # if 40 < len(contours[index]) < 100:
#     #     #     temp = cv.drawContours(thresh, contours, index, 145, 3)
#
#     smoothened = []
#     for contour in contours:
#         x, y = contour.T
#
#         x = x.tolist()[0]
#         y = y.tolist()[0]
#
#         tck, u = splprep([x, y], u=None, s=1.0, per=1)
#
#         u_new = np.linspace(u.min(), u.max(), 25)
#
#         x_new, y_new = splev(u_new, tck, der=0)
#
#         res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
#         smoothened.append(np.asarray(res_array, dtype=np.int32))
#
#         # approx = cv.approxPolyDP(smoothened, 0.01 * cv.arcLength(smoothened, True), True)
#
#         temp = cv.drawContours(temp, smoothened, -1, 255, 2)
#
#     return temp
#
#
# def find_clubhead(frame):
#     result = kmeans_clustering(frame)
#     # cv.imshow("Cluserting", result)
#     contours = bin_threshold(result)
#     # cv.imshow("Contours", contours)
#
#     return contours
#
#
# def optical_flow(frame, is_colour):
#     grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     temp_frame = frame.copy()
#
#     corners = find_corners(grayframe)
#     clubhead = find_clubhead(frame)
#     # clubhead = np.uint16(np.around(clubhead))
#     #
#     # for i in clubhead[0, :]:
#     #     # draw the outer circle
#     #     cv.circle(temp_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
#     #     # draw the center of the circle
#     #     cv.circle(temp_frame, (i[0], i[1]), 2, (0, 0, 255), 3)
#     for i in corners:
#         x, y = i.ravel()
#         cv.circle(temp_frame, (x, y), 3, 255, -1)
#
#     # cv.imshow("Detected Clubhead", clubhead)
#     # cv.imshow("Frame", frame)
#     # cv.waitKey(0)
#     return temp_frame
#
#
# def background_subtraction(frame, is_colour):
#     """
#     Execute background substitution on the frame
#     :return: retrun the resulting frame
#     """
#     # Assign vars and pick background subtraction method
#     g_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#
#     # Apply background substitution
#     fg_mask = subtraction.apply(g_frame)
#     ret, fg_mask = cv.threshold(fg_mask, 127, 255, cv.THRESH_BINARY)
#
#     # cv.imshow("Foreground Mask", fg_mask)
#     if is_colour:
#         return cv.bitwise_and(frame, frame, mask=fg_mask)
#     else:
#         return cv.bitwise_and(g_frame, fg_mask)
#
#
# def main_loop():
#     while capture.isOpened():
#         ret, frame = capture.read()
#         comp_ret, comp_frame = compare_capture.read()
#         comp_frame = comp_frame[0:1200, 0:600]
#         # if frame is None:
#         #     break
#
#         # if frame is read correctly ret is True
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break
#
#         frame = cv.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)))
#         frame = cv.rotate(frame, cv.ROTATE_180)
#
#         temp_frame = frame.copy()
#         # fg_frame = background_subtraction(temp_frame, True)
#         # cv.imshow('Foreground', fg_frame)
#         # cv.imshow("Foreground frame", of_frame)
#
#         # of_frame = optical_flow(frame, True)
#         # of_frame = optical_flow(temp_frame, True)
#         #
#         # cv.imshow('Optical Flow Frame', of_frame)
#         # hands = hand_estimation(temp_frame)
#         # # comp_hands = hand_estimation(comp_frame)
#         # cv.imshow('Hand estimation', hands)
#
#         pose_frame = pose_estimation(temp_frame)
#         cv.imshow("Pose Estimation", pose_frame)
#
#         # temp_com = centre_of_mass(comp_frame)
#
#         # com_loc = centre_of_mass(temp_frame)
#         # player_centre = cv.circle(frame, (np.uint8(np.ceil(com_loc["c_x"])), np.uint8(np.ceil(com_loc["c_y"]))), 1,
#         #                           (0, 255, 255), 5)
#
#         cv.imshow('Frame', frame)
#         # cv.imshow('Compare Frame', comp)
#
#         # print(capture.get(1))
#         if (capture.get(1) == 50.0) or (capture.get(1) == 75.0) or (capture.get(1) == 80.0):
#             print("yep")
#             cv.waitKey(500000)
#
#         keyboard = cv.waitKey(30)
#         if keyboard == 'q' or keyboard == 27:
#             break
#
#
# if __name__ == "__main__":
#     # play the video, main loop
#     main_loop()
import sys

import cv2 as cv

# Config Class Imports
import my_config.config as config
# Custom class imports
from helper_funcs.analyse_swing import SwingImageAnalyser
from helper_funcs.ball_detector import detect
from helper_funcs.classify_stage import StageClassifier
from helper_funcs.data_record import DTLDataRecord as DTLData
from helper_funcs.data_record import FODataRecord as FOData
from helper_funcs.feedback_system import NegativeFeedback as Feedback
from helper_funcs.graphing import GraphHelper as Graph
from helper_funcs.pose_estimation import PoseEstimation as Pose

# Retrieve input file from run-time ram
# parser = argparse.ArgumentParser(description='This program shows how to improve a golf swing using OpenCV methods.')
# parser.add_argument('--input', type=str, help='Path to a video.',
#                     default="./videos/samples/sample0.mov")
# parser.add_argument('--input1', type=str, help='Path to a video.',
#                     default="./videos/samples/sample7.mov")
# args = parser.parse_args()

print("Retrieiving video from files")

# frontal_view = cv.VideoCapture(args.input)
# dtl_view = cv.VideoCapture(args.input1)
# compare_capture = cv.VideoCapture("./videos/tiger.mp4")

# if not frontal_view.isOpened() or not dtl_view.isOpened():
#     print("Input file failed to open, there is a file path error.")
#     sys.exit(0)
#
# # Pose estimator
pose = Pose()
#
# # Drawing functionality
draw = Graph()
#
# # Data recorded
fo_data = FOData()
dtl_data = DTLData()
#
# # TODO: Get rid of not here only used for dev
isMac = config.isMac()
#
temp = 0

# Debug help
# pause_at = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
#             30, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 65, 66, 67, 68, 69, 70, 75, 76]
# pause_at = [6, 12, 32, 46, 67]  # rough classifications


pause_at = [6, 20, 38, 48, 50, 51, 75]
# pause_at = [9, 21, 41, 47, 48, 53, 80]
pause_at = []


def face_on_view():
    # TODO: Delete this and replace with uploaded video
    frontal_view = cv.VideoCapture("./FO-video.MOV")

    if not frontal_view.isOpened():
        print("Input file failed to open, there is a file path error.")
        sys.exit(0)

    # Save the analysed video
    fps = int(frontal_view.get(cv.CAP_PROP_FPS))
    width = int(frontal_view.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(frontal_view.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'avc1')  # this fourcc works form chrome and firefox
    out = cv.VideoWriter('./video/FO_analysed.mp4', fourcc, fps, (int(width / 2.5), int(height / 2.5)))

    slomo = False
    print("Processing the inputted video")
    while frontal_view.isOpened():
        if slomo:
            cv.waitKey(200)
        ret, frame = frontal_view.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Get frame number
        frame_num = frontal_view.get(cv.CAP_PROP_POS_FRAMES)
        print("Frame{}".format(str(frame_num)))

        if frame_num in pause_at:
            cv.waitKey()

        frame = cv.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
        global temp
        temp = frame.shape[1]

        if not isMac:
            frame = cv.rotate(frame, cv.ROTATE_180)

        # Get pose estimation for frame
        # whole_pose = pose.predict_pose(frame)
        # draw.draw_pose(frame, whole_pose)

        # Segmented video
        # mask = pose.segmentation(frame)
        # try:
        #     cv.imshow('Segmentation', mask.segmentation_mask)
        # except Exception:
        #     print("Exception in mask print")

        # segmentation = pose.segmentation(frame)

        ball = detect(frame)
        # Get relevant pose features
        results = pose.predict_relevant(frame)
        draw.draw_pose_results(frame, results)

        checks = draw.draw_expanded(frame, results, ball)

        # # TODO: Fix this
        # temp = FOData()
        # fo_data = temp
        fo_data.store_frame_data(results)

        # buffer = cv.imencode('.jpg', checks)[1].tobytes()
        # print("Sending frame.jpg to client")
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')

        # temp = detect(frame)
        out.write(checks)

        # TODO: Parse video to drop irrelevant frames

        # TODO: Classify swing stage (downswing, followthrough etc...)
        # Draw frame number
        text = f'Frame: {int(frame_num)}'
        cv.putText(frame, text, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # cv.imshow('Frame', frame)
        keyboard = cv.waitKey(1)
        # keyboard = cv.waitKey(100)
        width = frame.shape[1]
        height = frame.shape[0]

        # if keyboard == 115:  # 115 is "s"
        #     slomo = not slomo
        #     cv.waitKey(1000)
        #     print(slomo)
        # if keyboard == 113:  # 113 is "q"
        #     sys.exit(0)

    out.release()
    cv.destroyAllWindows()
    return fo_data


def down_the_line():
    # TODO: Delete this and replace with uploaded video
    dtl_view = cv.VideoCapture("./DTL-video.MOV")

    if not dtl_view.isOpened():
        print("Input file failed to open, there is a file path error.")
        sys.exit(0)

    # Save the analysed video
    fps = int(dtl_view.get(cv.CAP_PROP_FPS))
    width = int(dtl_view.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(dtl_view.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'avc1')  # this fourcc works form chrome and firefox
    out = cv.VideoWriter('./video/DTL_analysed.mp4', fourcc, fps, (int(width / 2.5), int(height / 2.5)))

    # Process Down the Line view
    slomo = False
    while dtl_view.isOpened():
        if slomo:
            cv.waitKey(200)
        ret, frame = dtl_view.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_num = dtl_view.get(cv.CAP_PROP_POS_FRAMES)
        print("Frame{}".format(str(frame_num)))

        if frame_num in pause_at:
            cv.waitKey()

        frame = cv.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
        if not isMac:
            frame = cv.rotate(frame, cv.ROTATE_180)

        ball = detect(frame)

        # Get relevant pose features
        # TODO update this to only fill in NONE where the landmark is NONE (instead of the entire result)
        # try:
        #     results = pose.predict_dtl_pose(frame)
        # except:
        #     print("Pose results were NONE on frame")
        results = pose.predict_dtl_pose(frame)
        draw.draw_pose_results(frame, results)

        checks = draw.draw_dtl_checks(frame, results, ball)

        out.write(checks)

        dtl_data.store_frame_data(results)

        # TODO: Parse video to drop irrelevant frames

        # cv.imshow('Frame', frame)
        keyboard = cv.waitKey(1)
        if keyboard == 115:  # 115 is "s"
            slomo = not slomo
            cv.waitKey(10000)
            print(slomo)

        if keyboard == 113:  # 113 is "q"
            sys.exit(0)
    cv.destroyAllWindows()
    return dtl_data


# def main_loop():
#     face_on_view()
#     down_the_line()


if __name__ == "__main__":
    # Synchornise the 2 angles of the videos.
    # TODO: Uncomment this and change to parser not sync
    # sync = Synchronizer(frontal_view, dtl_view)
    # sync.main()

    # frontal_view = cv.VideoCapture('videos/synced/synced-a.mp4')
    # dtl_view = cv.VideoCapture('videos/synced/synced-b.mp4')

    # play the video, main loop
    # main_loop()
    face_on_view()
    down_the_line()
    # TODO: Graph tracking at end of video
    draw.show_graphs(fo_data, dtl_data, temp)

    # Get the acceleration curves for the different views for the stage classification
    acc = draw.get_processed_fo_data()
    acc2 = draw.get_processed_dtl_data()
    print("Video is being classified for its stages")
    classifier = StageClassifier(acc, acc2, frontal_view, dtl_view)
    classifier.classify()

    # TODO: Analyse these relevant "frames" and preform checks
    print("Analysing the inputted video for corrections")
    swing_analyser = SwingImageAnalyser()
    analysed_res = swing_analyser.analyse()
    # for item in results:
    #     print("In {}".format(item["Stage"]))
    #     print(item["Description"])
    #     if item["isMistake"]:
    #         print("This will effect {}".format(item["Problem"]))
    #     else:
    #         print("This will help {}".format(item["Problem"]))
    feedback = Feedback(analysed_res)
    feedback.process()

    # Results contains all the data we need for the front end. Only additional thing needed it the photos/video.
    # When we have this we can use results["stage"] to get the photo and draw the points and whatever else necessary
