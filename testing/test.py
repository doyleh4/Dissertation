"""
This script provides the testing functionality for the project.

This test script will use a pre-selected video and using the manually 
selected ground-truth points will determine the accuracy. 

Debug is a boolean that will provide visual results for each test frame
"""
# Necessary
import cv2 as cv
import numpy as np

# Custom classes
from helper_funcs.pose_estimation import PoseEstimation as Pose

debug = True

# Test video
# video_path = "../videos/sample/sample6.mov"
# cap = cv.VideoCapture(video_path)

# directory paths
frame_dir = "./ground_truth/frames"
ground_truth_dir = "./ground_truth/truths"

# left_wrist_truth_path = "./ground_truth/left_wrist"
# left_shoulder_truth_path = "./ground_truth/left_shoulder"

# Frames we have ground truth for
# The ground truth images MUST be named by these indices
frame_indices = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70]

# Pose class
pose = Pose()

# def test_left_wrist(frame, index):
#     wrist_coords = pose.get_left_wrist(frame)
#
#     # Get ground truth image for this frame
#     ground_truth = cv.imread("{}/{}.jpg".format(left_wrist_truth_path, str(index)))
#
#     # Get a binary image for the marked ground truth area
#     lower_red = np.array([0, 0, 250])
#     higher_red = np.array([5, 5, 255])
#     mask = cv.inRange(ground_truth, lower_red, higher_red)
#
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv.dilate(mask, kernel, iterations=2)
#
#     print("For frame {} value at coordinates is {}".format(index, mask[wrist_coords[1], wrist_coords[0]]))
#     # if
#
#     if debug:
#         temp = ground_truth.copy()
#         cv.circle(temp, wrist_coords, 3, 175, -1)
#         cv.imshow("Ground Truth", temp)
#         cv.waitKey()
#     return 0


# def test_left_shoulder(frame, index):
#     shoulder_coords = pose.get_left_shoulder(frame)
#
#     # Get ground truth image for this frame
#     ground_truth = cv.imread("{}/{}.jpg".format(left_shoulder_truth_path, str(index)))
#
#     # Get a binary image for the marked ground truth area
#     lower_red = np.array([0, 0, 250])
#     higher_red = np.array([5, 5, 255])
#     mask = cv.inRange(ground_truth, lower_red, higher_red)
#
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv.dilate(mask, kernel, iterations=2)
#
#     print("For frame {} value at coordinates is {}".format(index, mask[shoulder_coords[1], shoulder_coords[0]]))
#     # if
#
#     if debug:
#         temp = ground_truth.copy()
#         cv.circle(temp, shoulder_coords, 3, (0, 175, 175), -1)
#         cv.imshow("Ground Truth", temp)
#         cv.waitKey()


LEFT_WRIST_COLOUR_LOW = [0, 0, 250]  # bgr
LEFT_WRIST_COLOUR_UP = [5, 5, 255]  # bgr


def test_left_wrist(img, ground_truth):
    coords = pose.get_left_wrist(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= LEFT_WRIST_COLOUR_LOW) and np.all(colour <= LEFT_WRIST_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()

    # If we decide to use assertions this is how.    print("oop") will ruin if assert error occurs
    # assert np.all(colour >= LEFT_WRIST_COLOUR_LOW) or np.all(np.all(colour <= LEFT_WRIST_COLOUR_UP)), print("oop")


LEFT_ELBOW_COLOUR_LOW = [0, 95, 195]  # bgr
LEFT_ELBOW_COLOUR_UP = [5, 102, 202]  # bgr


def test_left_elbow(img, ground_truth):
    coords = pose.get_left_elbow(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= LEFT_ELBOW_COLOUR_LOW) and np.all(colour <= LEFT_ELBOW_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


LEFT_SHOULDER_COLOUR_LOW = [147, 147, 147]  # bgr
LEFT_SHOULDER_COLOUR_UP = [152, 152, 152]  # bgr


def test_left_shoulder(img, ground_truth):
    coords = pose.get_left_shoulder(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= LEFT_SHOULDER_COLOUR_LOW) and np.all(colour <= LEFT_SHOULDER_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


LEFT_KNEE_COLOUR_LOW = [250, 0, 0]  # bgr
LEFT_KNEE_COLOUR_UP = [255, 5, 5]  # bgr


def test_left_knee(img, ground_truth):
    coords = pose.get_left_knee(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= LEFT_KNEE_COLOUR_LOW) and np.all(colour <= LEFT_KNEE_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


LEFT_HIP_COLOUR_LOW = [146, 146, 0]  # bgr
LEFT_HIP_COLOUR_UP = [152, 152, 5]  # bgr


def test_left_hip(img, ground_truth):
    coords = pose.get_left_hip(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]
    # print("Colour: {}".format(colour))

    if np.all(colour >= LEFT_HIP_COLOUR_LOW) and np.all(colour <= LEFT_HIP_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


LEFT_ANKLE_COLOUR_LOW = [250, 146, 0]  # bgr
LEFT_ANKLE_COLOUR_UP = [255, 152, 5]  # bgr


def test_left_ankle(img, ground_truth):
    coords = pose.get_left_ankle(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]
    # print("Colour: {}".format(colour))

    if np.all(colour >= LEFT_ANKLE_COLOUR_LOW) and np.all(colour <= LEFT_ANKLE_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_WRIST_COLOUR_LOW = [120, 146, 40]  # bgr
RIGHT_WRIST_COLOUR_UP = [127, 152, 47]  # bgr


def test_right_wrist(img, ground_truth):
    coords = pose.get_right_wrist(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_WRIST_COLOUR_LOW) and np.all(colour <= RIGHT_WRIST_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_ELBOW_COLOUR_LOW = [120, 146, 142]  # bgr
RIGHT_ELBOW_COLOUR_UP = [127, 152, 147]  # bgr


def test_right_elbow(img, ground_truth):
    coords = pose.get_right_elbow(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_ELBOW_COLOUR_LOW) and np.all(colour <= RIGHT_ELBOW_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_SHOULDER_COLOUR_LOW = [0, 146, 142]  # bgr
RIGHT_SHOULDER_COLOUR_UP = [5, 152, 147]  # bgr


def test_right_shoulder(img, ground_truth):
    coords = pose.get_right_shoulder(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_SHOULDER_COLOUR_LOW) and np.all(colour <= RIGHT_SHOULDER_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_HIP_COLOUR_LOW = [0, 66, 142]  # bgr
RIGHT_HIP_COLOUR_UP = [5, 72, 147]  # bgr


def test_right_hip(img, ground_truth):
    coords = pose.get_right_hip(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_HIP_COLOUR_LOW) and np.all(colour <= RIGHT_HIP_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_KNEE_COLOUR_LOW = [197, 67, 250]  # bgr
RIGHT_KNEE_COLOUR_UP = [202, 72, 255]  # bgr


def test_right_knee(img, ground_truth):
    coords = pose.get_right_knee(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_KNEE_COLOUR_LOW) and np.all(colour <= RIGHT_KNEE_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_ANKLE_COLOUR_LOW = [197, 147, 250]  # bgr
RIGHT_ANKLE_COLOUR_UP = [202, 152, 255]  # bgr


def test_right_ankle(img, ground_truth):
    coords = pose.get_right_ankle(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_ANKLE_COLOUR_LOW) and np.all(colour <= RIGHT_ANKLE_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_HEEL_COLOUR_LOW = [47, 197, 250]  # bgr
RIGHT_HEEL_COLOUR_UP = [52, 202, 255]  # bgr


def test_right_heel(img, ground_truth):
    coords = pose.get_right_heel(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_HEEL_COLOUR_LOW) and np.all(colour <= RIGHT_HEEL_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_TOE_COLOUR_LOW = [0, 250, 0]  # bgr
RIGHT_TOE_COLOUR_UP = [5, 255, 5]  # bgr


def test_right_toe(img, ground_truth):
    coords = pose.get_right_toe(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_TOE_COLOUR_LOW) and np.all(colour <= RIGHT_TOE_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


NOSE_COLOUR_LOW = [72, 250, 0]  # bgr
NOSE_COLOUR_UP = [78, 255, 5]  # bgr


def test_nose(img, ground_truth):
    coords = pose.get_nose(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= NOSE_COLOUR_LOW) and np.all(colour <= NOSE_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


EAR_COLOUR_LOW = [0, 250, 250]  # bgr
EAR_COLOUR_UP = [5, 255, 255]  # bgr


def test_left_ear(img, ground_truth):
    coords = pose.get_left_ear(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= EAR_COLOUR_LOW) and np.all(colour <= EAR_COLOUR_UP):
        # TODO: Increment true positive
        print("Oh its a hit")
    else:
        # TODO: increment false positive
        print("miss")

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


if __name__ == "__main__":
    # print("Running tests for the project")
    # # TODO: Change this to be for the given images instead of the video feed
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #
    #     # Resize and rotate
    #     # TODO: Again once i figure this out remove
    #     frame = cv.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
    #     frame = cv.rotate(frame, cv.ROTATE_180)
    #
    #     # if frame is read correctly ret is True
    #     if not ret:
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break
    #
    #     frame_index = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    #
    #     if frame_index in frame_indices:
    #         # print("Frame to test occcured: Running Tests")
    #         # test_left_wrist(frame, frame_index)
    #         test_left_shoulder(frame, frame_index)
    print("Running Tests")

    for frame_index in frame_indices:
        img = cv.imread("{}/{}.jpg".format(frame_dir, frame_index))
        ground_truth_img = cv.imread("{}/{}.jpg".format(ground_truth_dir, frame_index))
        #
        test_left_wrist(img, ground_truth_img)
        test_left_elbow(img, ground_truth_img)
        test_left_shoulder(img, ground_truth_img)
        test_left_knee(img, ground_truth_img)
        test_left_hip(img, ground_truth_img)
        test_left_ankle(img, ground_truth_img)
        test_right_wrist(img, ground_truth_img)
        test_right_elbow(img, ground_truth_img)
        test_right_shoulder(img, ground_truth_img)
        test_right_hip(img, ground_truth_img)
        test_right_knee(img, ground_truth_img)
        test_right_ankle(img, ground_truth_img)
        test_right_heel(img, ground_truth_img)
        test_right_toe(img, ground_truth_img)
        test_nose(img, ground_truth_img)
        test_left_ear(img, ground_truth_img)
