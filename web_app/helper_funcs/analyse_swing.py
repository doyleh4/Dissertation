# Required imports
# Helper import
import cv2 as cv
import numpy as np

# Custom class imports
from web_app.helper_funcs.ball_detector import detect
from web_app.helper_funcs.graphing import GraphHelper as Graph
from web_app.helper_funcs.pose_estimation import AnalysePose

SETUP_FRAME = ("swing_stages/face_on/setup.jpg", "swing_stages/dtl/setup.jpg")
TAKEAWAY_FRAME = ("swing_stages/face_on/takeaway.jpg", "swing_stages/dtl/takeaway.jpg")
BACKSWING_FRAME = ("swing_stages/face_on/backswing.jpg", "swing_stages/dtl/backswing.jpg")
DOWNSWING_FRAME = ("swing_stages/face_on/downswing.jpg", "swing_stages/dtl/downswing.jpg")
IMPACT_FRAME = ("swing_stages/face_on/post-impact.jpg", "swing_stages/dtl/post-impact.jpg")
# TODO: Add post impact and interpolate between values
FOLLOWTHROUGH_FRAME = ("swing_stages/face_on/followthrough.jpg", "swing_stages/dtl/followthrough.jpg")

STAGE_PATH = [SETUP_FRAME, TAKEAWAY_FRAME, BACKSWING_FRAME, DOWNSWING_FRAME, IMPACT_FRAME, FOLLOWTHROUGH_FRAME]

draw = Graph()

setup_knee_angle = 0

global res
res = []


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
    return (b[1] - a[1]) / (b[0] - a[0])


def verify_leg_width(pose, img):
    """
    Function to verify that the legs are shoulder with apart
    :param pose:
    :param img:
    :return:
    """
    check = {
        "Check": "Legs Shoulder Width Apart",
        "Stage": "Setup",
        "Problem": "Balance",
        "Description": "",  # Description of what's being done in the swing
        "Fix": "",  # Filled in by the advice feedback system
        "Points": [],
        # all these below used as metadata
        "isMistake": False,
        "isRootCause": True,  # if this mistake leads to another we will need to check for the other aswell
        "LeadsTo": ["Back Foot on Toes", "Head Over Ball", "Knee Flex Maintained for Setup"],
        "isProcessed": False  # once check has been done, mark as true
    }

    ankles = pose.get_ankles()
    shoulders = pose.get_shoulders()

    check["Points"].append(ankles)
    check["Points"].append(shoulders)

    # Draw the check
    draw.leg_width_check(img, ankles, shoulders)

    # Calculate the pixel distances
    # Positive means AHEAD, negative means BEHIND
    left = ankles[0][0] - shoulders[0][0]
    right = ankles[1][0] - shoulders[1][0]

    if -5 > left > 5 and -5 > right > 5:
        check["Description"] += "One or both of your shoulders is not directly above your leg"
        check["isMistake"] = True
    else:
        check["Description"] += "Your shoulders and feet were properly lined up in setup"

    res.append(check)


def verify_one_piece_movement(pose, img):
    """
    Verify the users arms and shoulders are moving as one
    :param pose:
    :param img:
    :return:
    """
    check = {
        "Check": "One Piece Movement",
        "Stage": "Takeaway",
        "Problem": "Rotation Axis",
        "Description": "",  # Description of what's being done in the swing
        "Fix": "",  # Filled in by the advice feedback system
        "Points": [],
        # all these below used as metadata
        "isMistake": False,
        "isRootCause": True,  # if this mistake leads to another we will need to check for the other aswell
        "LeadsTo": ["Lead Arm on Shoulder Plane", "Closed Shoulders"],
        "isProcessed": False  # once check has been done, mark as true
    }

    wrists = pose.get_wrists()
    elbows = pose.get_elbows()
    shoulders = pose.get_shoulders()

    check["Points"].append(wrists)
    check["Points"].append(elbows)
    check["Points"].append(shoulders)

    draw.one_piece_movement_check(img, wrists, elbows, shoulders)

    # Check angle at each elbow. Should be nearly 0 to mark one-piece
    left_side_angle = calculate_angle(elbows[0], wrists[0], shoulders[0])
    right_side_angle = calculate_angle(elbows[1], wrists[1], shoulders[1])

    # TODO: Verify the angle here is working as expected. (is it the inside or outside angle)
    if not 170 < left_side_angle < 190 and not 170 < right_side_angle < 190:  # 10 degree margin for error
        check["Description"] += "One piece movement did not occur. Your shoulders, elbows and wrists should all move" \
                                " as one in order to maintain rotation axis for the movement"
        check["isMistake"] = True
    else:
        check["Description"] += "One Piece Movement happened as expected. This will help maintain rotation axis " \
                                "throughout the movement"

    res.append(check)


def verify_followthrough_checks(pose, img):
    check = {
        "Check": "Head over Lead Leg",
        "Stage": "Followthrough",
        "Problem": "Rotation Axis",
        "Description": "",  # Description of what's being done in the swing
        "Fix": "",  # Filled in by the advice feedback system
        "Points": [],
        # all these below used as metadata
        "isMistake": False,
        "isRootCause": False,  # if this mistake leads to another we will need to check for the other aswell
        "LeadsTo": [],
        "isProcessed": False  # once check has been done, mark as true
    }
    right_shoulder = pose.get_shoulders()[1]
    left_ankle = pose.get_ankles()[0]

    check["Points"].append(right_shoulder)
    check["Points"].append(left_ankle)

    draw.shoulder_over_foot(img, right_shoulder, left_ankle)

    distance = right_shoulder[0] - left_ankle[0]

    if distance > 5:  # Margin for acceptance
        check["Description"] += "Your right shoulder is infront of your left ankle, this shows rotation axis was " \
                                "misaligned during the movement and balance was off "
        check["isMistake"] = True
    elif distance < -5:  # Margin for acceptance
        check["Description"] += "Your right shoulder is behind of your left ankle, this shows rotation axis was " \
                                "misaligned during the movement and balance was off "
        check["isMistake"] = True
    else:
        check[
            "Description"] += "Your trail shoulder ended over your lead foot, this shows balance was maintained " \
                              "throughout the action and rotation axis remained stable "

    res.append(check)
    # TODO: Add in the check to see if the trail foot is lifted off the ground.


def verify_head_behind_ball(pose, img):
    check = {
        "Check": "Head Behind Ball",
        "Stage": "Downswing",
        "Problem": "Proper contact",
        "Description": "",  # Description of what's being done in the swing
        "Fix": "",  # Filled in by the advice feedback system
        "Points": [],
        # all these below used as metadata
        "isMistake": False,
        "isRootCause": False,  # if this mistake leads to another we will need to check for the other aswell
        "LeadsTo": [],
        "isProcessed": False  # once check has been done, mark as true
    }
    # ball = detect(img)
    ball = list(detect(img)[0])
    head = pose.get_left_ear()

    check["Points"].append(ball)
    check["Points"].append(head)

    # draw.head_behind_ball(img, ball, head)

    # Head should be behind the golf ball, hence subtract x values and if positive good
    diff = head[0] - ball[0]
    if diff < -5:  # allow slight margin as
        check["Description"] += "Your head is not adequately behind the golf ball, this will may prevent more solid " \
                                "contact. "
        check["isMistake"] = True
    else:
        check["Description"] += "Your head remained behind the golf ball allowing for solid contact"

    res.append(check)


def verify_knee_angle(pose, img, name):
    check = {
        "Check": "Knee Flex {}".format(name),
        "Stage": name,
        "Problem": "Balance",
        "Description": "",  # Description of what's being done in the swing
        "Fix": "",  # Filled in by the advice feedback system
        "Points": [],
        # all these below used as metadata
        "isMistake": False,
        "isRootCause": True,  # if this mistake leads to another we will need to check for the others
        "LeadsTo": None,
        "isProcessed": False  # once check has been done, mark as true
    }
    if name == "Setup":
        check["LeadsTo"] = ["Back Foot on Toes", "Head over Lead Leg", "Knee Flex {}".format("Downswing")]
    elif name == "Downswing":
        check["LeadsTo"] = ["Back Foot on Toes", "Head over Lead Leg"]

    ankle = pose.get_right_ankle()
    knee = pose.get_right_knee()
    hip = pose.get_right_hip()

    check["Points"].append(ankle)
    check["Points"].append(knee)
    check["Points"].append(hip)

    angle = calculate_angle(knee, ankle, hip)

    # TODO add angle check feedback. NOTE: Remember this is used in the downswing check aswell, so have it work for both
    draw.knee_angle(img, ankle, knee, hip, name)

    if 150 < angle < 160:
        check["Description"] += "Knee angle is setup to maintain balance. "
    elif angle < 150:
        check["Description"] += "Your knee is bent too much, this may effect your balance in the movement. "
        check["isMistake"] = True
    elif angle > 160:
        check["Description"] += "Your knee is not bent enough, this may effect your balance in the movement. "
        check["isMistake"] = True

    # Also as part of this we need to check that the users knee is directly above the foot. (not forward or backward)
    temp = [knee[0], knee[1] + 10]  # Get a point directly below the knee
    angle = calculate_angle(knee, ankle, temp)

    if -2 < angle < 2:
        check["Description"] += "Your knee is directly above your ankle, this will help to maintain balance. "
    elif angle < -2:
        check[
            "Description"] += "Your knee is too far ahead of your ankle, this may effect your balance in the movement. "
        check["isMistake"] = True
    elif angle > 2:
        check["Description"] += "Your knee is behind your ankle, this may effect your balance in the movement. "
        check["isMistake"] = True

    res.append(check)  # Both views


def verify_trail_arm_straight(pose, dtl_img):
    check = {
        "Check": "Trail Arm Straight",
        "Stage": "Takeaway",
        "Problem": "Consistency",
        "Description": "",  # Description of what's being done in the swing
        "Fix": "",  # Filled in by the advice feedback system
        "Points": [],
        # all these below used as metadata
        "isMistake": False,
        "isRootCause": False,  # if this mistake leads to another we will need to check for the others
        "LeadsTo": [],
        "isProcessed": False  # once check has been done, mark as true
    }

    wrist = pose.get_right_wrist()
    elbow = pose.get_right_elbow()
    shoulder = pose.get_right_shoulder()

    check["Points"].append(wrist)
    check["Points"].append(elbow)
    check["Points"].append(shoulder)

    angle = calculate_angle(elbow, shoulder, wrist)

    draw.trail_arm_straight(dtl_img, wrist, elbow, shoulder)

    if 175 < angle < 185:
        check["Description"] += "Your trail arm was straight help to lead reduced wrist usage and inconsistency"
    else:
        check["Description"] += "Your trail arm was not straight. This may lead to over usage of the wrists throughout " \
                                "the movement which will reduce consistency between swings"
        check["isMistake"] = True

    res.append(check)


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
    verify_knee_angle(pose, dtl_img, "Setup")


def run_takeaway_checks(fo_img, dtl_img):
    pose = AnalysePose(fo_img)

    verify_one_piece_movement(pose, fo_img)

    pose = AnalysePose(dtl_img)
    verify_trail_arm_straight(pose, dtl_img)
    # Verify clubhead out of line of hands??


def verify_shoulder_slope(pose, dtl_img):
    check = {
        "Check": "Lead Arm on Shoulder Plane",
        "Stage": "Backswing",
        "Problem": "Axis Rotation",
        "Description": "",  # Description of what's being done in the swing
        "Fix": "",  # Filled in by the advice feedback system
        "Points": [],
        # all these below used as metadata
        "isMistake": False,
        "isRootCause": True,  # if this mistake leads to another we will need to check for the others
        "LeadsTo": ["Closed Shoulders"],
        "isProcessed": False  # once check has been done, mark as true
    }
    shoulders = pose.get_shoulders()
    arm = [pose.get_left_wrist(), pose.get_left_elbow()]

    check["Points"].append(shoulders)
    check["Points"].append(arm)

    slope = calculate_slope(shoulders[0], shoulders[1])
    slope2 = calculate_slope(arm[0], arm[1])
    draw.shoulder_slope(dtl_img, shoulders, arm)

    if slope2 - .35 < slope < slope2 + .35:
        check["Description"] += "Your shoulders and lead arm are on the same plane in the backswing, this will enable " \
                                "you to stay on the correct rotation axis for the movement"
    else:
        check["Description"] += "Your shoulders and lead arm are not in the same plane, this will ofset the rotation " \
                                "axis for the movement"
        check["isMistake"] = True

    res.append(check)


def verify_elbow_pointing_down(pose, dtl_img):
    check = {
        "Check": "Trail Elbow in Right Direction",
        "Stage": "Backswing",
        "Problem": "Consistency",
        "Description": "",  # Description of what's being done in the swing
        "Fix": "",  # Filled in by the advice feedback system
        "Points": [],
        # all these below used as metadata
        "isMistake": False,
        "isRootCause": False,  # if this mistake leads to another we will need to check for the others
        "LeadsTo": [],
        "isProcessed": False  # once check has been done, mark as true
    }
    elbow = pose.get_right_elbow()
    wrist = pose.get_right_wrist()

    check["Points"].append(elbow)
    check["Points"].append(wrist)

    slope = calculate_slope(elbow, wrist)

    draw.elbow_pointing_down(dtl_img, elbow, wrist)

    if slope < 0:
        check["Description"] += "Lead elbow pointing down and to the left, this will aid in being more consistent " \
                                "between your swings"
    else:
        check["Description"] += "Your lead elbow is not pointing down and to the left, this may affect consistency " \
                                "between swings"
        check["isMistake"] = True

    res.append(check)


def verify_shoulders_closed(pose, img):
    check = {
        "Check": "Closed Shoulders",
        "Stage": "Downswing",
        "Problem": "Rotation Axis",
        "Description": "",  # Description of what's being done in the swing
        "Fix": "",  # Filled in by the advice feedback system
        "Points": [],
        # all these below used as metadata
        "isMistake": False,
        "isRootCause": False,  # if this mistake leads to another we will need to check for the others
        "LeadsTo": [],
        "isProcessed": False  # once check has been done, mark as true
    }
    shoulders = pose.get_shoulders()

    check["Points"].append(shoulders)

    slope = calculate_slope(shoulders[0], shoulders[1])

    # Add check for slope value that shows slightly closed (negative but approx 0)
    draw.shoulders_closed(img, shoulders)

    if 0.1 < slope < 0.3:
        check["Description"] += "Your shoulders are closed properly, indicating the swing stayed on plane"
    elif slope < 0.1:
        check["Description"] += "Your shoulders were open, this shows the swing is off plane"
        check["isMistake"] = True
    elif slope > 0.3:
        check["Description"] += "Your shoulders were too far closed, this shows the swing is off plane"
        check["isMistake"] = True

    res.append(check)


def run_backswing_checks(fo_img, dtl_img):
    # Add in a way to check the back is facing the target

    # Down the line checks
    pose = AnalysePose(dtl_img)

    verify_shoulder_slope(pose, dtl_img)
    verify_elbow_pointing_down(pose, dtl_img)


def run_downswing_checks(fo_img, dtl_img):
    pose = AnalysePose(fo_img)

    # TODO: Uncomment when we get a new video with a ball
    # verify_head_behind_ball(pose, fo_img)

    # Down the line checks
    pose = AnalysePose(dtl_img)
    verify_knee_angle(pose, dtl_img, "Downswing")
    verify_shoulders_closed(pose, dtl_img)


def run_impact_checks(fo_img, dtl_img):
    # Interpolate between these points
    pass


def run_followthrough_checks(fo_img, dtl_img):
    pose = AnalysePose(fo_img)

    verify_followthrough_checks(pose, fo_img)
    # Note: We do not need to preform the DTL checks here as the check is the same as one of the FO checks


class SwingImageAnalyser:
    """
        Class to provide SwingAnalysis when we have the images not video. This will also operate on neutral
        feedback loop, telling the user what is wrong in their swing. This will be used as input into the
        advice system.
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
            # # TODO: Can we scale this to fit on the monitor as opposed to resizing image
            # fo_img = cv.resize(fo_img, (int(fo_img.shape[1] / 2.5), int(fo_img.shape[0] / 2.5)))
            # dtl_img = cv.resize(dtl_img, (int(dtl_img.shape[1] / 2.5), int(dtl_img.shape[0] / 2.5)))
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

        return res


def run_dtl_setup_checks(dtl_img):  # img and frame here are interchangeable
    """
    Function to run the checks to be carried out in the swing setup.
    :param fo_img, dtl_img:
    :return:
    """
    # Carry out down-the-line checks
    pose = AnalysePose(dtl_img)

    # Verify the knee is slightly bent
    verify_knee_angle(pose, dtl_img, "Setup")


def run_dtl_takeaway_checks(dtl_img):
    pose = AnalysePose(dtl_img)
    verify_trail_arm_straight(pose, dtl_img)
    # Verify clubhead out of line of hands??


def run_dtl_backswing_checks(dtl_img):
    # Down the line checks
    pose = AnalysePose(dtl_img)

    verify_shoulder_slope(pose, dtl_img)
    verify_elbow_pointing_down(pose, dtl_img)


def run_dtl_downswing_checks(dtl_img):
    pose = AnalysePose(dtl_img)
    verify_knee_angle(pose, dtl_img, "Downswing")
    verify_shoulders_closed(pose, dtl_img)


def run_dtl_followthrough_checks(dtl_img):
    # No DTL Checks to implement atm
    pass


class DTLAnalyser:
    def analyse(self):
        res.clear()
        setup_img = "./video/setup.jpg"
        takeaway_img = "./video/takeaway.jpg"
        backswing_img = "./video/backswing.jpg"
        downswing_img = "./video/downswing.jpg"
        impact_img = "./video/post-impact.jpg"
        # TODO: Add post impact and interpolate between values
        followthrough_img = "./video/followthrough.jpg"

        paths = [setup_img, takeaway_img, backswing_img, downswing_img, impact_img, followthrough_img]

        for dtl in paths:
            # Read image from opencv
            # try:
            dtl_img = cv.imread(dtl)

            # depending on what stage of the swing we are checking call different functions
            if dtl == setup_img:
                run_dtl_setup_checks(dtl_img)
            elif dtl == takeaway_img:
                run_dtl_takeaway_checks(dtl_img)
            elif dtl == backswing_img:
                run_dtl_backswing_checks(dtl_img)
            elif dtl == downswing_img:
                run_dtl_downswing_checks(dtl_img)
            # elif (face_on, dtl) == impact_img:
            #     # process the post impact frame aswell, may be better to do it here and avoid it being in the list
            #     run_impact_checks(img, dtl_img)
            elif dtl == followthrough_img:
                run_dtl_followthrough_checks(dtl_img)

        return res


def run_fo_setup_checks(img):
    # Uses the pose_estimation file, with the constructor for the analyse class
    pose = AnalysePose(img)

    # draw.draw_pose_results(fo_img, pose.results)
    # Verify the legs are shoulder width apart
    verify_leg_width(pose, img)


def run_fo_takeaway_checks(img):
    pose = AnalysePose(img)

    verify_one_piece_movement(pose, img)


def run_fo_backswing_checks(img):
    # No checks to do here?
    pass


def run_fo_downswing_checks(img):
    pose = AnalysePose(img)

    # TODO: Uncomment when we get a new video with a ball
    # verify_head_behind_ball(pose, img)


def run_fo_followthrough_checks(img):
    pose = AnalysePose(img)

    verify_followthrough_checks(pose, img)


class FOAnalyser:
    def analyse(self):
        res.clear()
        setup_img = "./video/setup.jpg"
        takeaway_img = "./video/takeaway.jpg"
        backswing_img = "./video/backswing.jpg"
        downswing_img = "./video/downswing.jpg"
        impact_img = "./video/post-impact.jpg"
        # TODO: Add post impact and interpolate between values
        followthrough_img = "./video/followthrough.jpg"

        paths = [setup_img, takeaway_img, backswing_img, downswing_img, impact_img, followthrough_img]

        for dtl in paths:
            # Read image from opencv
            # try:
            img = cv.imread(dtl)

            # depending on what stage of the swing we are checking call different functions
            if dtl == setup_img:
                run_fo_setup_checks(img)
            elif dtl == takeaway_img:
                run_fo_takeaway_checks(img)
            elif dtl == backswing_img:
                run_fo_backswing_checks(img)
            elif dtl == downswing_img:
                run_fo_downswing_checks(img)
            # elif (face_on, dtl) == impact_img:
            #     # process the post impact frame aswell, may be better to do it here and avoid it being in the list
            #     run_impact_checks(img, dtl_img)
            elif dtl == followthrough_img:
                run_fo_followthrough_checks(img)

        return res


# Next class - Used for analysing video
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
