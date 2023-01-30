import math


class DataRecord:
    """
        Class to provide data storage functionality
    """

    def __init__(self):
        self.data = {
            "lw": [],
            "shoulder_slope": [],
            "lead_leg_to_shoulder": [],
            "acc": []
        }

    def store_frame_data(self, pose_data):
        """
        Function to store all the recorded data into persistent storage.
        Only works with specific data (not entire pose set)
        :param pose_data:
        :return:
        """
        """
        Indexes - 0: left wrist, 1: right wrist, 2: left elbow, 3: right elbow, 4: left shoulder, 5: right shoulder,
        6: left hip, 7: right hip, 8: left knee, 9: right knee, ,10: left ankle, 11: right ankle, 12: right heel,
        13: right foot index/toe, 14: left ear, 15: nose
        """
        self.data["lw"].append(pose_data[0])
        self.data["shoulder_slope"].append((pose_data[5][1] - pose_data[4][1]) / (pose_data[5][0] - pose_data[4][0]))
        self.data["lead_leg_to_shoulder"].append(pose_data[11][0] - pose_data[5][0])

        if len(self.data["lw"]) > 1:
            dist = math.dist(self.data["lw"][len(self.data["lw"]) - 2], self.data["lw"][len(self.data["lw"]) - 1])
            self.data["acc"].append(dist)
            # self.data["acc"].append(
            #     # self.data["lw"][len(self.data["lw"]) - 2] - self.data["lw"][
            #     #     len(self.data["lw"]) - 1] / 0.6)  # formula for acceleration 0.6 is frame rate 60 fps
            #     math.sqrt()
            # )
