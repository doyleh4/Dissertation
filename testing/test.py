"""
This script provides the testing functionality for the project.

This test script will use a pre-selected video and using the manually 
selected ground-truth points will determine the accuracy. 

Debug is a boolean that will provide visual results for each test frame
"""
# Necessary
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Custom classes
from web_app.helper_funcs.pose_estimation import PoseEstimation as Pose

debug = False

# Test video
# video_path = "../video/sample/sample6.mov"
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

LEFT_WRIST_COLOUR_LOW = [0, 0, 250]  # bgr
LEFT_WRIST_COLOUR_UP = [5, 5, 255]  # bgr

# Left wrist
lw_t = 0
lw_f = 0


def test_left_wrist(img, ground_truth):
    global lw_t, lw_f
    coords = pose.get_left_wrist(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= LEFT_WRIST_COLOUR_LOW) and np.all(colour <= LEFT_WRIST_COLOUR_UP):
        lw_t += 1
    else:
        lw_f += 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()

    # If we decide to use assertions this is how.    print("oop") will ruin if assert error occurs
    # assert np.all(colour >= LEFT_WRIST_COLOUR_LOW) or np.all(np.all(colour <= LEFT_WRIST_COLOUR_UP)), print("oop")


LEFT_ELBOW_COLOUR_LOW = [0, 95, 195]  # bgr
LEFT_ELBOW_COLOUR_UP = [5, 102, 202]  # bgr

le_t = 0
le_f = 0


def test_left_elbow(img, ground_truth):
    global le_t, le_f
    coords = pose.get_left_elbow(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= LEFT_ELBOW_COLOUR_LOW) and np.all(colour <= LEFT_ELBOW_COLOUR_UP):
        # TODO: Increment true positive
        le_t += 1
    else:
        # TODO: increment false positive
        le_f += 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


LEFT_SHOULDER_COLOUR_LOW = [147, 147, 147]  # bgr
LEFT_SHOULDER_COLOUR_UP = [152, 152, 152]  # bgr

ls_t = 0
ls_f = 0


def test_left_shoulder(img, ground_truth):
    global ls_f, ls_t
    coords = pose.get_left_shoulder(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= LEFT_SHOULDER_COLOUR_LOW) and np.all(colour <= LEFT_SHOULDER_COLOUR_UP):
        # TODO: Increment true positive
        ls_t += 1
    else:
        # TODO: increment false positive
        ls_f = + 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


LEFT_KNEE_COLOUR_LOW = [250, 0, 0]  # bgr
LEFT_KNEE_COLOUR_UP = [255, 5, 5]  # bgr

lk_t = 0
lk_f = 0


def test_left_knee(img, ground_truth):
    global lk_t, lk_f
    coords = pose.get_left_knee(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= LEFT_KNEE_COLOUR_LOW) and np.all(colour <= LEFT_KNEE_COLOUR_UP):
        # TODO: Increment true positive
        lk_t += 1
    else:
        # TODO: increment false positive
        lk_f += 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


LEFT_HIP_COLOUR_LOW = [146, 146, 0]  # bgr
LEFT_HIP_COLOUR_UP = [152, 152, 5]  # bgr

lh_t = 0
lh_f = 0


def test_left_hip(img, ground_truth):
    global lh_t, lh_f
    coords = pose.get_left_hip(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]
    # print("Colour: {}".format(colour))

    if np.all(colour >= LEFT_HIP_COLOUR_LOW) and np.all(colour <= LEFT_HIP_COLOUR_UP):
        # TODO: Increment true positive
        lh_t += 1
    else:
        # TODO: increment false positive
        lh_f += 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


LEFT_ANKLE_COLOUR_LOW = [250, 146, 0]  # bgr
LEFT_ANKLE_COLOUR_UP = [255, 152, 5]  # bgr

la_t = 0
la_f = 0


def test_left_ankle(img, ground_truth):
    global la_t, la_f
    coords = pose.get_left_ankle(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]
    # print("Colour: {}".format(colour))

    if np.all(colour >= LEFT_ANKLE_COLOUR_LOW) and np.all(colour <= LEFT_ANKLE_COLOUR_UP):
        # TODO: Increment true positive
        la_t += 1
    else:
        # TODO: increment false positive
        la_f += 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_WRIST_COLOUR_LOW = [120, 146, 40]  # bgr
RIGHT_WRIST_COLOUR_UP = [127, 152, 47]  # bgr

rw_t = 0
rw_f = 0


def test_right_wrist(img, ground_truth):
    global rw_t, rw_f
    coords = pose.get_right_wrist(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_WRIST_COLOUR_LOW) and np.all(colour <= RIGHT_WRIST_COLOUR_UP):
        # TODO: Increment true positive
        rw_t += 1
    else:
        # TODO: increment false positive
        rw_f += 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_ELBOW_COLOUR_LOW = [120, 146, 142]  # bgr
RIGHT_ELBOW_COLOUR_UP = [127, 152, 147]  # bgr

re_t = 0
re_f = 0


def test_right_elbow(img, ground_truth):
    global re_t, re_f
    coords = pose.get_right_elbow(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_ELBOW_COLOUR_LOW) and np.all(colour <= RIGHT_ELBOW_COLOUR_UP):
        # TODO: Increment true positive
        re_t += 1
    else:
        # TODO: increment false positive
        re_f += 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_SHOULDER_COLOUR_LOW = [0, 146, 142]  # bgr
RIGHT_SHOULDER_COLOUR_UP = [5, 152, 147]  # bgr

rs_t = 0
rs_f = 0


def test_right_shoulder(img, ground_truth):
    global rs_f, rs_t
    coords = pose.get_right_shoulder(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_SHOULDER_COLOUR_LOW) and np.all(colour <= RIGHT_SHOULDER_COLOUR_UP):
        # TODO: Increment true positive
        rs_t += 1
    else:
        # TODO: increment false positive
        rs_f += 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_HIP_COLOUR_LOW = [0, 66, 142]  # bgr
RIGHT_HIP_COLOUR_UP = [5, 72, 147]  # bgr

rh_t = 0
rh_f = 0


def test_right_hip(img, ground_truth):
    global rh_f, rh_t
    coords = pose.get_right_hip(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_HIP_COLOUR_LOW) and np.all(colour <= RIGHT_HIP_COLOUR_UP):
        # TODO: Increment true positive
        rh_t += 1
    else:
        # TODO: increment false positive
        rh_f += 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_KNEE_COLOUR_LOW = [197, 67, 250]  # bgr
RIGHT_KNEE_COLOUR_UP = [202, 72, 255]  # bgr

rk_t = 0
rk_f = 0


def test_right_knee(img, ground_truth):
    global rk_t, rk_f
    coords = pose.get_right_knee(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_KNEE_COLOUR_LOW) and np.all(colour <= RIGHT_KNEE_COLOUR_UP):
        # TODO: Increment true positive
        rk_t += 1
    else:
        # TODO: increment false positive
        rk_f += 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_ANKLE_COLOUR_LOW = [197, 147, 250]  # bgr
RIGHT_ANKLE_COLOUR_UP = [202, 152, 255]  # bgr

ra_t = 0
ra_f = 0


def test_right_ankle(img, ground_truth):
    global ra_t, ra_f
    coords = pose.get_right_ankle(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_ANKLE_COLOUR_LOW) and np.all(colour <= RIGHT_ANKLE_COLOUR_UP):
        # TODO: Increment true positive
        ra_t += 1
    else:
        # TODO: increment false positive
        ra_f += 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_HEEL_COLOUR_LOW = [47, 197, 250]  # bgr
RIGHT_HEEL_COLOUR_UP = [52, 202, 255]  # bgr

rh1_t = 0
rh1_f = 0


def test_right_heel(img, ground_truth):
    global rh1_f, rh1_t
    coords = pose.get_right_heel(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_HEEL_COLOUR_LOW) and np.all(colour <= RIGHT_HEEL_COLOUR_UP):
        rh1_t += 1
    else:
        rh1_f += 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


RIGHT_TOE_COLOUR_LOW = [0, 250, 0]  # bgr
RIGHT_TOE_COLOUR_UP = [5, 255, 5]  # bgr

t_t = 0
t_f = 0


def test_right_toe(img, ground_truth):
    global t_t, t_f
    coords = pose.get_right_toe(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= RIGHT_TOE_COLOUR_LOW) and np.all(colour <= RIGHT_TOE_COLOUR_UP):
        t_t += 1
    else:
        t_f += 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


NOSE_COLOUR_LOW = [72, 250, 0]  # bgr
NOSE_COLOUR_UP = [78, 255, 5]  # bgr

n_t = 0
n_f = 0


def test_nose(img, ground_truth):
    global n_f, n_t
    coords = pose.get_nose(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= NOSE_COLOUR_LOW) and np.all(colour <= NOSE_COLOUR_UP):
        # TODO: Increment true positive
        n_t += 1
    else:
        # TODO: increment false positive
        n_f += 1

    if debug:
        temp = ground_truth.copy()

        cv.circle(temp, coords, 2, (255, 255, 255), -1)
        cv.imshow("Temp", temp)
        cv.waitKey()


EAR_COLOUR_LOW = [0, 250, 250]  # bgr
EAR_COLOUR_UP = [5, 255, 255]  # bgr

e_t = 0
e_f = 0


def test_left_ear(img, ground_truth):
    global e_t, e_f
    coords = pose.get_left_ear(img)

    # Check the colour of that coordinate in the ground truth image
    colour = ground_truth[coords[1], coords[0]]

    if np.all(colour >= EAR_COLOUR_LOW) and np.all(colour <= EAR_COLOUR_UP):
        e_t += 1
    else:
        # TODO: increment false positive
        e_f += 1

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

    labels = ["Left Wrist", "Left Elbow", "Left Shoulder", "Left Knee", "Left Hip", "Left Ankle", "Right Wrist",
              "Right Elbow", "Right Shoulder", "Right Knee", "Right Hip", "Right Ankle", "Right Toe", "Nose", "Ear"]

    true_items = [lw_t, le_t, ls_t, lk_t, lh_t, la_t, rw_t, re_t, rs_t, rk_t, rh_t, ra_t, t_t, n_t, e_t]
    false_items = [lw_f, le_f, ls_f, lk_f, lh_f, la_f, rw_f, re_f, rs_f, rk_f, rh_f, ra_f, t_f, n_f, e_f]

    l_width = 0.25
    plt.bar(labels, true_items, l_width, color="green", label="Correct")
    plt.bar([i + l_width for i in range(len(labels))], false_items, l_width, color="red", label="Incorrect")

    plt.title("Body Point Detection validation")
    plt.xticks(rotation=45)  # Rotate the labels so they can all fit
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.xlabel("Point")
    plt.ylabel("Results")
    plt.legend()

    plt.show()

    s = sum(true_items)
    f = sum(false_items)
    print("Recall: {}".format(str(s / (s + f))))
