# Functional requirements
import math

# Required imports
import cv2 as cv
import numpy as np
import tqdm
from matplotlib import pyplot as plt

import my_config.config as config
# Custom class imports
from helper_funcs.pose_estimation import PoseEstimation


def find_stop_point(arr):
    """
    Function to return a local minimum around maximum value
    """
    max_index = arr.argmax()

    for i in range(max_index, 0, -1):
        if arr[i] < arr[i - 1]:
            return i
    return max_index


def calculate_acceleration(filled):
    """
    TODO: This is similar to the method in data_record.py. Uses a different smoothening technique so make as 1 maybe
    """
    res = []
    for i in range(1, len(filled) - 1):
        try:
            dist = math.dist(filled[i - 1], filled[i])
            res.append(dist)
        except:
            print("None value encountered")

    # Smoothen this to make the classification easier
    # TODO: This savgol filer parameters could be changed if the frame rate is differnet (i.e. slo-mo camera), this result
    # doesnt turn out similar in slo-mo video
    # res = savgol_filter(res, 30, 3)  # window size 30, polynomial order 6 - good for regular speed
    # res = savgol_filter(res, 180, 3)  # (expected shape, but bumpy) - good for slo-mo but needs more experimentation
    # res = savgol_filter(res, 12, 3)

    kernel_size = 7
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved_10 = np.convolve(res, kernel, mode='same')

    return data_convolved_10


# Variable definitions
pose = PoseEstimation()
arr2 = []
isMac = config.isMac()


class Synchronizer:
    def __init__(self, face_on, dtl):
        self.fo_cap = face_on
        self.dtl_cap = dtl

        self.arr = []

    def initial_playthrough(self):
        """
        Initial play of the video to analyse feature positioning
        """
        while self.fo_cap.isOpened():
            ret, frame = self.fo_cap.read()
            # try:
            #     ret2, frame2 = self.dtl_cap.read()
            # except:
            #     print("second file not opened")

            # End play back if end of video
            if not ret:  # Need to uclude " or not ret2" in conditional:
                break

            frame = cv.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)

            # frame2 = cv.resize(frame2, (int(frame2.shape[1] / 2.5), int(frame2.shape[0] / 2.5)))
            # if not isMac:
            #     frame2 = cv.rotate(frame2, cv.ROTATE_180)

            try:
                self.arr.append(pose.get_left_wrist(frame))
                # arr2.append(pose.get_left_wrist(frame2))
            except:
                self.arr.append([None, None])
                # arr2.append([None, None])

            # cv.imshow("Face On", frame)
            # cv.imshow("Down The Line", frame2)
            # cv.waitKey(5)

        # Restart video
        self.fo_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        # self.dtl_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        cv.destroyAllWindows()

    def save_synced_videos(self, a, b):
        """
        Function to save the synced video;
        """
        print("Processing video. This can take upto a minute")

        # Set up video writer config
        width = int(self.fo_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.fo_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.fo_cap.get(cv.CAP_PROP_FPS))

        fourcc = cv.VideoWriter_fourcc(*'mp4v')

        # out = cv.VideoWriter("video/synced/synced-a.mp4", fourcc, fps, (width, height))
        out = cv.VideoWriter("video/temp_parsed.mp4", fourcc, fps, (width, height))
        # out2 = cv.VideoWriter("video/synced/synced-b.mp4", fourcc, fps, (width, height))

        # Get 25 frames either side of top of backswing from these indices
        print("First video")
        for i in tqdm.tqdm(range(a - 40, a + 40)):
            # Read and write video between these frames
            self.fo_cap.set(cv.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.fo_cap.read()
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            out.write(frame)

        # print("Second video")
        # for i in tqdm.tqdm(range(b - 40, b + 40)):
        #     # Read and write video between these frames
        #     self.dtl_cap.set(cv.CAP_PROP_POS_FRAMES, i)
        #     ret, frame = self.dtl_cap.read()
        #     if not isMac:
        #         frame = cv.rotate(frame, cv.ROTATE_180)
        #     out2.write(frame)

        out.release()
        # out2.release()
        self.fo_cap.release()
        return [a-40, a+40]
        # self.dtl_cap.release()

    def main(self):
        """
        Main function of the class. Will save the new synced video and return the new CV.VideoCapture objects
        """
        if not self.fo_cap.isOpened():  # Include this in conditional "or not self.dtl_cap.isOpened():"
            print('Error loading video file')
            exit()

        self.initial_playthrough()

        # Calculate acceleration curves
        acc = calculate_acceleration(self.arr)
        # acc2 = calculate_acceleration(arr2)

        # Plot the acceleration curve
        # i = np.arange(len(acc))
        # plt.plot(i, acc, label="Movement per frame")
        # plt.ylabel("Movement (pix/frame)")
        # plt.xlabel("Frame")
        # plt.legend()
        # plt.show()

        # Find end of backswing (when acceleration is zero)
        a = find_stop_point(acc)
        b = find_stop_point(np.array([0, 1]))  # find_stop_point(acc2)

        return self.save_synced_videos(a, b)

        # new_a = cv.VideoCapture('video/synced/synced-a.mp4')
        # new_b = cv.VideoCapture('video/synced/synced-b.mp4')
        #
        #
        # while new_a.isOpened() or new_b.isOpened():
        #     ret, frame = new_a.read()
        #     ret2, frame2 = new_b.read()
        #
        #     # End play back if end of video
        #     if not ret or not ret2:
        #         break
        #
        #     cv.imshow("Face On", frame)
        #     cv.imshow("Down The Line", frame2)
        #     cv.waitKey(5)


if __name__ == "__main__":
    temp = cv.VideoCapture("../video/temp.MOV")
    sync = Synchronizer(temp, None)
    sync.main()
