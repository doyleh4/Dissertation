# Required imports
# Helper import

import cv2 as cv
import numpy as np

from helper_funcs.ball_detector import detect
from helper_funcs.graphing import GraphHelper as Graph
# Custom class imports
from helper_funcs.pose_estimation import AnalysePose

SETUP_FRAME = ("swing_stages/face_on/setup.jpg", "swing_stages/dtl/setup.jpg")
TAKEAWAY_FRAME = ("swing_stages/face_on/takeaway.jpg", "swing_stages/dtl/takeaway.jpg")
BACKSWING_FRAME = ("swing_stages/face_on/backswing.jpg", "swing_stages/dtl/backswing.jpg")
DOWNSWING_FRAME = ("swing_stages/face_on/downswing.jpg", "swing_stages/dtl/downswing.jpg")
IMPACT_FRAME = ("swing_stages/face_on/post-impact.jpg", "swing_stages/dtl/post-impact.jpg")
# TODO: Add post impact and interpolate between values
FOLLOWTHROUGH_FRAME = ("swing_stages/face_on/followthrough.jpg", "swing_stages/dtl/followthrough.jpg")

STAGE_PATH = [SETUP_FRAME, TAKEAWAY_FRAME, BACKSWING_FRAME, DOWNSWING_FRAME, IMPACT_FRAME, FOLLOWTHROUGH_FRAME]

draw = Graph()


# TODO: Change all measurements here to be body to pixel ratio.
# TODO: Add docstring everwhere here
# TODO: Change all print() to save to a textfile to be fed into feedback system one by one

def calculate_angle(main, a, b):
    # Calcaute the vectors
    v1 = a - main
    v2 = b - main

    # Calculate and return angle
    return np.degrees(np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2)))

def calculate_slope(a, b):
    """ Calcuates the slope from a -> b"""
    return (b[1]-a[1])/(b[0]-a[0])

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
    left_side_angle = calculate_angle(elbows[0], wrists[0], shoulders[0])
    right_side_angle = calculate_angle(elbows[1], wrists[1], shoulders[1])

    # TODO: Verify the angle here is working as expected. (is it the inside or outside angle)
    if not 170 < left_side_angle < 190 and not 170 < right_side_angle < 190:  # 10 degree margin for error
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


def verify_knee_angle(pose, img, name):
    ankle = pose.get_right_ankle()
    knee = pose.get_right_knee()
    hip = pose.get_right_hip()

    # TODO: Verify this angle is being done correctly
    angle = calculate_angle(knee, hip, ankle)

    # TODO add angle check feedback. NOTE: Remember this is used in the downswing check aswell, so have it work for both

    draw.knee_angle(img, ankle, knee, hip, name)


def verify_trail_arm_straight(pose, dtl_img):
    wrist = pose.get_right_wrist()
    elbow = pose.get_right_elbow()
    shoulder = pose.get_right_shoulder()

    angle = calculate_angle(elbow, shoulder, wrist)

    draw.trail_arm_straight(dtl_img, wrist, elbow, shoulder)


def run_setup_checks(fo_img, dtl_img):  # img and frame here are interchangeable
    """
    Function to run the checks to be carried out in the swing setup.
    :param fo_img, dtl_img:
    :return:
    """
    # Uses the pose_estimation file, with the constructor for the analyse class
    pose = AnalysePose(fo_img)
    # print(pose.results)

    # draw.draw_pose_results(fo_img, pose.results)
    # Verify the legs are shoulder width apart
    verify_leg_width(pose, fo_img)

    # Carry out down-the-line checks
    pose = AnalysePose(dtl_img)

    # Verify the knee is slightly bent 
    verify_knee_angle(pose, dtl_img, "setup")
    # TODO: Verify the back is straight


def run_takeaway_checks(fo_img, dtl_img):
    pose = AnalysePose(fo_img)

    verify_one_piece_movement(pose, fo_img)

    pose = AnalysePose(dtl_img)
    verify_trail_arm_straight(pose, dtl_img)
    # Verify clubhead out of line of hands??


def verify_shoulder_slope(pose, dtl_img):
    # TODO: This check isnt shoulder slope, its to check if the left arm is ion the same plane as the shoulders
    #  so add this to the check
    shoulders = pose.get_shoulders()

    slope = calculate_slope(shoulders[0], shoulders[1])
    draw.shoulder_slope(dtl_img, shoulders)


def verify_elbow_pointing_down(pose, dtl_img):
    elbow = pose.get_right_elbow()
    shoulder = pose.get_right_shoulder()

    slope = calculate_slope(elbow, shoulder)

    draw.elbow_pointing_down(dtl_img, elbow, shoulder)

    # TODO: This simply checks if its pointing down, change it so it checks pointing down and left (probable a range of vals)
    if slope < 0:
        print("Good: Right elbow pointing down and to the left")
    else:
        print("In the backswing the players left elbow is not pointing down and to the left")


def verify_shoulders_closed(pose, img):
    shoulders = pose.get_shoulders()

    slope = calculate_slope(shoulders[0], shoulders[1])

    # Add check for slope value that shows slightly closed (negative but approx 0)
    draw.shoulders_closed(img, shoulders)


def run_backswing_checks(fo_img, dtl_img):
    # Add in a way to check the back is facing the target

    # Down the line checks
    pose = AnalysePose(dtl_img)

    verify_shoulder_slope(pose, dtl_img)
    verify_elbow_pointing_down(pose, dtl_img)
    pass


def run_downswing_checks(fo_img, dtl_img):
    pose = AnalysePose(fo_img)

    # verify_head_behind_ball(pose, fo_img)

    #Down the line checks
    pose = AnalysePose(dtl_img)
    verify_knee_angle(pose, dtl_img, "downswing")
    verify_shoulders_closed(pose, dtl_img)


def run_impact_checks(fo_img, dtl_img):
    # Interpolate between these points
    pass


def run_followthrough_checks(fo_img, dtl_img):
    pose = AnalysePose(fo_img)

    verify_followthrough_checks(pose, fo_img)

    # TODO: The above is only to check if the head is above the left foot, add in the right foot of the ground from
    #  both views?


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

        for face_on, dtl in STAGE_PATH:
            # Read image from opencv
            # try:
            fo_img = cv.imread(face_on)
            dtl_img = cv.imread(dtl)

            # Show image and scale size to screen. Going to be different between my Mac and Windows. So fix this, but
            # doesn't matter to the functionality at the end.
            # TODO: Can we scale this to fit on the monitor as opposed to resizing image
            fo_img = cv.resize(fo_img, (int(fo_img.shape[1] / 2.5), int(fo_img.shape[0] / 2.5)))
            dtl_img = cv.resize(dtl_img, (int(dtl_img.shape[1] / 2.5), int(dtl_img.shape[0] / 2.5)))
            # cv.imshow('image', img)

            # except:
            #     print("Error reading image")

            # depending on what stage of the swing we are checking call different functions
            if (face_on, dtl) == SETUP_FRAME:
                run_setup_checks(fo_img, dtl_img)
            elif (face_on, dtl) == TAKEAWAY_FRAME:
                run_takeaway_checks(fo_img, dtl_img)
            elif (face_on, dtl) == BACKSWING_FRAME:
                run_backswing_checks(fo_img, dtl_img)
            elif (face_on, dtl) == DOWNSWING_FRAME:
                run_downswing_checks(fo_img, dtl_img)
            # elif (face_on, dtl) == IMPACT_FRAME:
            #     # process the post impact frame aswell, may be better to do it here and avoid it being in the list
            #     run_impact_checks(img, dtl_img)
            elif (face_on, dtl) == FOLLOWTHROUGH_FRAME:
                run_followthrough_checks(fo_img, dtl_img)

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
