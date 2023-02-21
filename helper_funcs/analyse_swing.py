# Required imports
# Helper import

import cv2 as cv
import numpy as np

from helper_funcs.ball_detector import detect
from helper_funcs.graphing import GraphHelper as Graph
# Custom class imports
from helper_funcs.pose_estimation import AnalysePose

SETUP_FRAME = "swing_stages/setup.jpg"
TAKEAWAY_FRAME = "swing_stages/takeaway.jpg"
BACKSWING_FRAME = "swing_stages/backswing.jpg"
# TODO: Add downswing vlassification
DOWNSWING_FRAME = "swing_stages/downswing.jpg"
IMPACT_FRAME = "swing_stages/post-impact.jpg"
# TODO: Add post impact and interpolate between values
FOLLOWTHROUGH_FRAME = "swing_stages/followthrough.jpg"

STAGE_PATH = [SETUP_FRAME, TAKEAWAY_FRAME, BACKSWING_FRAME, DOWNSWING_FRAME, IMPACT_FRAME, FOLLOWTHROUGH_FRAME]

draw = Graph()


# TODO: Change all measurements here to be body to pixel ratio.

# TODO: Change all print() to save to a textfile to be fed into feedback system one by one

def verify_leg_width(pose, img):
    """
    Function to verify that the legs are shoulder with apart
    :param pose:
    :param img:
    :return:
    """
    ankles = pose.get_ankles()
    shoulders = pose.get_shoulders()

    # Draw the check
    draw.leg_width_check(img, ankles, shoulders)

    # Calculate the pixel distances
    # Positive means AHEAD, negative means BEHIND
    left = ankles[0][0] - shoulders[0][0]
    right = ankles[1][0] - shoulders[1][0]

    if left > 5:  # Margin for acceptance
        print("In setup users left ankle is {} pixels infront of their shoulders".format(str(left)))
    elif left < -5:  # Margin for acceptance
        print("In setup users left ankle is {} pixels behind of their shoulders".format(str(left)))

    if right > 5:  # Margin for acceptance
        print("In setup users right ankle is {} pixels infront of their shoulders".format(str(right)))
    elif right < -5:  # Margin for acceptance
        print("In setup users right ankle is {} pixels behind of their shoulders".format(str(right)))


def calculate_angle(main, a, b):
    # Calcaute the vectors
    v1 = a - main
    v2 = b - main

    # Calculate and return angle
    return np.degrees(np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2)))


def verify_one_piece_movement(pose, img):
    """
    Verify the users arms and shoulders are moving as one
    :param pose:
    :param img:
    :return:
    """
    wrists = pose.get_wrists()
    elbows = pose.get_elbows()
    shoulders = pose.get_shoulders()

    draw.one_piece_movement_check(img, wrists, elbows, shoulders)

    # Check angle at each elbow. Should be nearly 0 to mark one-piece
    left_side_angle = calculate_angle(wrists[0], elbows[0], shoulders[0])
    right_side_angle = calculate_angle(wrists[1], elbows[1], shoulders[1])

    if -2 < left_side_angle < 0 and 0 < right_side_angle < 1:
        print("In takeaway shoulders, wrists and elbows did not move as one")
    else:
        print("Good: One Piece Movement was correct")


def verify_followthrough_checks(pose, img):
    right_shoulder = pose.get_shoulders()[1]
    left_ankle = pose.get_ankles()[0]

    draw.shoulder_over_foot(img, right_shoulder, left_ankle)

    distance = right_shoulder[0] - left_ankle[0]

    if distance > 5:  # Margin for acceptance
        print("In follow through users right shoulder is {} pixels infront of their left ankle".format(str(distance)))
    elif distance < -5:  # Margin for acceptance
        print("In follow through users right shoulder is {} pixels behind of their left ankle".format(str(distance)))
    else:
        print("Good: Balance was maintained in follow through for the shoulder over foot check")


def verify_head_behind_ball(pose, img):
    # ball = detect(img)
    ball = list(detect(img)[0])
    head = pose.get_left_ear()

    # draw.head_behind_ball(img, ball, head)

    # Head should be behind the golf ball, hence subtract x values and if positive good
    diff = head[0] - ball[0]
    if diff < -5:  # allow slight margin as
        print("In the downswing the users head is behind the golf ball by {} pixels".format(str(diff)))
    else:
        print("Good: Players head remained behind the golf ball allowing for solid contact")


def run_setup_checks(img):  # img and frame here are interchangeable
    """
    Function to run the checks to be carried out in the swing setup.
    :param img:
    :return:
    """
    # Uses the pose_estimation file, with the constructor for the analyse class
    pose = AnalysePose(img)
    # print(pose.results)

    draw.draw_pose_results(img, pose.results)
    # Verify the legs are shoulder width apart
    verify_leg_width(pose, img)


def run_takeaway_checks(img):
    pose = AnalysePose(img)

    verify_one_piece_movement(pose, img)


def run_backswing_checks(img):
    # Add in a way to check the back is facing the target
    pass


def run_impact_checks(img):
    # Interpolate between these points
    pass


def run_followthrough_checks(img):
    pose = AnalysePose(img)

    verify_followthrough_checks(pose, img)


def run_downswing_checks(img):
    pose = AnalysePose(img)

    verify_head_behind_ball(pose, img)


class SwingImageAnalyser:
    """
        Class to provide SwingAnalysis when we have the images not video
    """

    # def __init__(self):

    def analyse(self):
        """
        Main function of this class. Will iterate though the folder classified stage frames and analsyse them
        individually.
        """

        for path in STAGE_PATH:
            # Read image from opencv
            # try:
            img = cv.imread(path)

            # Show image and scale size to screen. Going to be different between my Mac and Windows. So fix this, but
            # doesn't matter to the functionality at the end.
            # TODO: Can we scale this to fit on the monitor as opposed to resizing image
            img = cv.resize(img, (int(img.shape[1] / 2.5), int(img.shape[0] / 2.5)))
            # cv.imshow('image', img)

            # except:
            #     print("Error reading image")

            # depending on what stage of the swing we are checking call different functions
            if path is SETUP_FRAME:
                run_setup_checks(img)
            elif path is TAKEAWAY_FRAME:
                run_takeaway_checks(img)
            elif path is BACKSWING_FRAME:
                run_backswing_checks(img)
            elif path is DOWNSWING_FRAME:
                run_downswing_checks(img)
            # elif path is IMPACT_FRAME:
            #     # process the post impact frame aswell, may be better to do it here and avoid it being in the list
            #     run_impact_checks(img)
            elif path is FOLLOWTHROUGH_FRAME:
                run_followthrough_checks(img)

        return 0


# Next class - Used for analysing videos
# TODO: implement this
class SwingVideoAnalyser:
    """
        Class to provide SwingAnalysis when we have input as video
    """

    # def __init__(self):

    def analyse(self):
        """
        """

        return 0
