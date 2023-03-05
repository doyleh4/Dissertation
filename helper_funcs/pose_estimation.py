# required imports
import cv2
import mediapipe as mp
import numpy as np


def normalise(x, y, width, height):
    return [int(x * width), int(y * height)]


def normalise_all(landmarks_subset, frame):
    points = []
    for landmark in landmarks_subset:
        # for each landmark get its normalised x, y coordinate
        normalised = normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])
        points.append(np.array(normalised))
    return points


class PoseEstimation:
    """
        Class to do pose estimation using MediaPipe
    """

    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(smooth_landmarks=True, min_tracking_confidence=0.75)

    def init_vars(self, frame):
        return frame.copy(), list(), self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def predict_pose(self, frame):
        """
            Will return a list of identified landmarks as co-ordinates
            :param frame:
            :return points:
        """
        res, points, results = self.init_vars(frame)

        landmarks = results.pose_landmarks

        return normalise_all(landmarks, frame)

    # TODO: Instead of single pose landmarks, return the landmarks (that matter) as a line of acceptance
    def predict_relevant(self, frame):
        """
            Will return a list of specific landmarks as co-ordinates
            :param frame:
            :return points:
        """
        res, points, results = self.init_vars(frame)

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
        # landmarks_subset = []
        # landmark_indexs = [15, 16, 13, 14, 11, 12, 23, 24, 25, 26, 27, 28, 30, 32, 7, 0]
        #
        # for index in landmark_indexs:
        #     if type(results.pose_landmarks.landmark[index]) is None:
        #         print("Mamma Mia")

        return normalise_all(landmarks_subset, frame)

    def predict_dtl_pose(self, frame):
        res, points, results = self.init_vars(frame)

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

        return normalise_all(landmarks_subset, frame)

    # TODO: Get mask of person segmentation and return that
    def segmentation(self, frame):
        seg_pose = self.mpPose.Pose(enable_segmentation=True)
        frame_small = frame.copy()
        frame_small = cv2.resize(frame_small, (frame_small.shape[1], frame_small.shape[0]))
        results = seg_pose.process(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))

        annotated_image = frame_small.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(frame_small.shape, dtype=np.uint8)
        bg_image[:] = (0, 0, 0)  # gray
        annotated_image = np.where(condition, annotated_image, bg_image)

        cv2.imshow("Segmentation", annotated_image)
        cv2.waitKey()

        return annotated_image

    # TODO: Could change this to be less repetitive (i.e. calculate the pose once)

    def get_left_wrist(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[15]  # left wrist

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_left_shoulder(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[11]  # left shoulder

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_left_elbow(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[13]  # left elbow

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_left_knee(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[25]  # left knee

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_left_hip(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[23]  # left hip

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_left_ankle(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[27]  # left ankle

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_right_wrist(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[16]  # right wrist

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_right_elbow(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[14]  # right elbow

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_right_shoulder(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[12]  # right shoulder

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_right_hip(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[24]  # right hip

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_right_knee(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[26]  # right knee

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_right_ankle(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[28]  # right ankle

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_right_heel(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[30]  # right heel

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_right_toe(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[32]  # right heel

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_nose(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[0]  # right heel

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])

    def get_left_ear(self, frame):
        res, points, results = self.init_vars(frame)
        landmark = results.pose_landmarks.landmark[7]  # right heel

        # Normalise co-ordinates and return
        return normalise(landmark.x, landmark.y, frame.shape[1], frame.shape[0])


class AnalysePose:
    """
    Helper for Pose analysis. Making it a separate class as we want to pass the frame in the contractor. Probably
    similar to above and should be made as a second constructor but don't think that would work
    """

    def __init__(self, frame):
        self.frame = frame

        mpPose = mp.solutions.pose
        pose = mpPose.Pose(smooth_landmarks=True, min_tracking_confidence=0.75)

        # Temp is the media pipe results
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

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

        # List of all pose points to use. In functions of this class just pick from this list
        self.results = normalise_all(landmarks_subset, frame)

    def get_ankles(self):
        """
        Get the 2 ankle coordinatess from the normalised list
        :return [left, right]:
        """
        return [self.results[10], self.results[11]]

    def get_shoulders(self):
        """
        Get the 2 shoulder coordinates from the normalised list
        :return [left, right]:
        """
        return [self.results[4], self.results[5]]

    def get_wrists(self):
        """
        Get the 2 shoulder coordinates from the normalised list
        :return [left, right]:
        """
        return [self.results[0], self.results[1]]

    def get_elbows(self):
        """
        Get the 2 shoulder coordinates from the normalised list
        :return [left, right]:
        """
        return [self.results[2], self.results[3]]

    def get_left_ear(self):
        return self.results[14]

    def get_right_ankle(self):
        return self.results[11]

    def get_right_knee(self):
        return self.results[9]

    def get_right_hip(self):
        return self.results[7]

    def get_right_wrist(self):
        return self.results[1]

    def get_right_elbow(self):
        return self.results[3]

    def get_right_shoulder(self):
        return self.results[5]

    def get_left_wrist(self):
        return self.results[0]

    def get_left_elbow(self):
        return self.results[2]
