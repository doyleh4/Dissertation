# required imports
import mediapipe as mp
import numpy as np


def normalise(x, y, width, height):
    return [int(x * width), int(y * height)]


class PoseEstimation:
    """
        Class to do pose estimation using MediaPipe
    """

    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

    def predict_pose(self, frame):
        """
        Will return a list of identified landmarks as co-ordinates
        :param frame:
        :return points : as np.array of normalised [[x,y]...]:
        """
        res = frame.copy()
        points = list()
        results = self.pose.process(frame)
        # landmarks = landmark_pb2.NormalizedLandmarkList(landmark=results.pose_landmarks.landmark)

        for landmark in results.pose_landmarks.landmark:
            # for each landmark get its normalised x, y coordinate
            # normed = landmark_pb2.NormalizedLandmark(landmark)
            normalised = normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])
            points.append(np.array(normalised))

        return points

    # TODO: Instead of single pose landmarks, return the landmarks (that matter) as a line of acceptance

    # TODO: Get mask of person segmentation and return that
