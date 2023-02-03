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
video_path = "../videos/sample/sample6.mov"
cap = cv.VideoCapture(video_path)

# Ground truth director
left_wrist_truth_path = "./ground_truth/left_wrist"

# Frames we have ground truth for
# The ground truth images MUST be named by these indices
frame_indices = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70]

# Pose class
pose = Pose()


def test_left_wrist(frame, index):
    wrist_coords = pose.get_left_wrist(frame)

    # Get ground truth image for this frame
    ground_truth = cv.imread("{}/{}.jpg".format(left_wrist_truth_path, str(index)))

    # Get a binary image for the marked ground truth area
    lower_red = np.array([0, 0, 250])
    higher_red = np.array([5, 5, 255])
    mask = cv.inRange(ground_truth, lower_red, higher_red)

    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv.dilate(mask, kernel, iterations=5)

    print("For frame {} value at coordinates is {}".format(index, mask[wrist_coords[1], wrist_coords[0]]))
    # if

    if debug:
        temp = ground_truth.copy()
        cv.circle(temp, wrist_coords, 3, (0, 175, 175), -1)
        cv.imshow("Ground Truth", temp)
        cv.waitKey()
    return 0


if __name__ == "__main__":
    print("Running tests for the project")
    # TODO: Change this to be for the given images instead of the video feed
    while cap.isOpened():
        ret, frame = cap.read()

        # Resize and rotate
        # TODO: Again once i figure this oput remove
        frame = cv.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
        frame = cv.rotate(frame, cv.ROTATE_180)

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_index = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        if frame_index in frame_indices:
            # print("Frame to test occcured: Running Tests")
            test_left_wrist(frame, frame_index)
