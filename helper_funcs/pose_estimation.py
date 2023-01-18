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
            :return points:
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
    def predict_relevant(self, frame):
        """
            Will return a list of specific landmarks as co-ordinates
            :param frame:
            :return points:
        """
        res = frame.copy()
        points = list()
        results = self.pose.process(frame)

        # Get the subset of landmarks that we need for the problem
        landmarks_subset = [
            results.pose_landmarks.landmark[15],  # 0: left wrist
            results.pose_landmarks.landmark[16],  # 1: right wrist
            results.pose_landmarks.landmark[13],  # 2: left elbow
            results.pose_landmarks.landmark[14],  # 3: right elbow
            results.pose_landmarks.landmark[11],  # 4: left shoulder
            results.pose_landmarks.landmark[12],  # 5: right shoulder
            results.pose_landmarks.landmark[23],  # 6: left hip
            results.pose_landmarks.landmark[24],  # 7: right hip
            results.pose_landmarks.landmark[25],  # 8: left knee
            results.pose_landmarks.landmark[26],  # 9: right knee
            results.pose_landmarks.landmark[27],  # 10: left ankle
            results.pose_landmarks.landmark[28],  # 11: right ankle
            results.pose_landmarks.landmark[30],  # 12: right heel
            results.pose_landmarks.landmark[32],  # 13: right foot index/toe
            results.pose_landmarks.landmark[7],  # 14: left ear
            results.pose_landmarks.landmark[0]  # 15: nose
        ]

        for landmark in landmarks_subset:
            # for each landmark get its normalised x, y coordinate
            normalised = normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])
            points.append(np.array(normalised))

        return points

    def predict_dtl_pose(self, frame):
        res = frame.copy()
        points = list()
        results = self.pose.process(frame)

        # Get the subset of landmarks that we need for the problem
        landmarks_subset = [
            results.pose_landmarks.landmark[24],  # 0: right hip
            results.pose_landmarks.landmark[26],  # 1: right knee
            results.pose_landmarks.landmark[28],  # 2: right ankle
            results.pose_landmarks.landmark[12],  # 3: right shoulder
            results.pose_landmarks.landmark[14],  # 4: right elbow
            results.pose_landmarks.landmark[16],  # 5: right wrist
            results.pose_landmarks.landmark[11],  # 6: left shoulder
            results.pose_landmarks.landmark[23],  # 7: left hip
            results.pose_landmarks.landmark[30],  # 8: right heel
            results.pose_landmarks.landmark[32],  # 9: right foot index/toe
        ]

        for landmark in landmarks_subset:
            # for each landmark get its normalised x, y coordinate
            normalised = normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])
            points.append(np.array(normalised))

        return points

    # TODO: Get mask of person segmentation and return that
