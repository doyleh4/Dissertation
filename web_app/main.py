# Required imports
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
#                     default="./video/samples/sample0.mov")
# parser.add_argument('--input1', type=str, help='Path to a video.',
#                     default="./video/samples/sample7.mov")
# args = parser.parse_args()

print("Retrieiving video from files")
# # Pose estimator
pose = Pose()
#
# # Drawing functionality
draw = Graph()
#
# # Data recorded
# fo_data = FOData()
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


def face_on_view(path):
    fo_data = FOData()
    # TODO: Delete this and replace with uploaded video
    frontal_view = cv.VideoCapture(path)

    if not frontal_view.isOpened():
        print("Input file failed to open, there is a file path error.")
        sys.exit(0)

    # Save the analysed video
    fps = int(frontal_view.get(cv.CAP_PROP_FPS))
    width = int(frontal_view.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(frontal_view.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # this fourcc works form chrome and firefox
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
        # cv.imshow("f", frame)

        # if keyboard == 115:  # 115 is "s"
        #     slomo = not slomo
        #     cv.waitKey(1000)
        #     print(slomo)
        # if keyboard == 113:  # 113 is "q"
        #     sys.exit(0)

    out.release()
    cv.destroyAllWindows()
    return fo_data


def down_the_line(path):
    # TODO: Delete this and replace with uploaded video
    dtl_view = cv.VideoCapture(path)

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
    # Synchornise the 2 angles of the video.
    # TODO: Uncomment this and change to parser not sync
    # sync = Synchronizer(frontal_view, dtl_view)
    # sync.main()

    # frontal_view = cv.VideoCapture('video/synced/synced-a.mp4')
    # dtl_view = cv.VideoCapture('video/synced/synced-b.mp4')

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
