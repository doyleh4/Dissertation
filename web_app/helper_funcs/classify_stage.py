import math
import os

import cv2 as cv
import my_config.config as config
import numpy as np
from scipy.signal import savgol_filter

"""
NCode used for testing but not used in the end
        ## Using the slope to classify the satge isnt great, better to use the apex of curve
        # slopes = []
        # for i in range(1, len(acc)):
        #     slopes.append((acc[i] - acc[i - 1]) / (y[i] - y[i - 1]))
        #
        # y = np.arange(len(slopes))
        # plt.plot(y, slopes)
        # plt.show()
"""
# TODO: Get rid of not here only used for dev
isMac = config.isMac()

SETUP = 1
TAKEAWAY = 2
BACKSWING = 3
DOWNSWING = 4
FOLLOWTHROUGH = 5

FO = "face_on"
DTL = "dtl"


def smoothen_curve(filled):
    # Smoothen this to make the classification easier# doesnt turn out similar in slo-mo video
    res = savgol_filter(filled, 31, 3)  # window size 31, polynomial order 3 - good for regular speed

    kernel_size = 9
    kernel = np.ones(kernel_size) / kernel_size
    res = np.convolve(res, kernel, mode='same')
    return res


def calc(data):
    res = []
    for i in range(1, len(data) - 1):
        try:
            dist = math.dist(data[i - 1], data[i])
            res.append(dist)
        except:
            print("None value encountered")
    return res


class StageClassifier:
    def __init__(self, data, video_feed):
        """
         Used in single processing on front-end
        :param data:
        :param video_feed:
        """
        self.data = data
        self.video = video_feed

        self.video.set(cv.CAP_PROP_POS_FRAMES, 0)

    def single_classify(self):
        """
        Used in single processing on front-end
        :return:
        """
        acc = smoothen_curve(self.data)

        self.classify_stages(acc, self.video)  # pass face on data and save

    def classify_stages(self, data, video):
        # NOTE: When we are saving the images here, the imwrite() function extracts the frame from the video
        # hence we need to take this into account when indexing the video after an imwrite(). Hence the index - states
        # when saving frames
        arr = []
        index = 0
        state = 0
        print(os.getcwd())
        current_trend = "down"
        # Get setup by getting the lowest acceleration point at the start
        if state == 0:
            for i in range(1, len(data)):
                if data[i] > data[i - 1]:
                    current_trend = "up"
                    index = i
                    arr.append(index)
                    break  # This is the first min
            # print("Low index is {}".format(str(index)))

            # Save this frame as an image
            video.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = video.read()
            frame = cv.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
            # NOTE: Drop "static in file name if not running website. Only needed in flask
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('video/setup.jpg', frame)
            state += 1

        # Get takeaway by detecting first significant change in the data values.
        if state == 1:
            temp = data[index:index + 25]
            start_point = np.array([index + 0, temp[0]])
            end_point = np.array([index + np.argmax(temp), np.max(temp)])

            diff = end_point[0] - start_point[0]
            point = start_point[0] + ((4 / 5) * diff)  # Get the 4/5 marker

            index = int(point)  # get index of this point
            arr.append(index)

            # Save this frame as an image
            video.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = video.read()
            frame = cv.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('video/takeaway.jpg', frame)
            state += 1

        # Get the top of the backswing by detecting the next low apex
        # Note: Due to the smoothening of the graph we need to offset this by roughly 5 frames
        current_trend = "up"
        if state == 2:
            prev_index = index
            change_count = 0

            for i in range(index, len(data)):
                if data[i] > data[i - 1]:
                    if current_trend is not "up":
                        current_trend = "up"
                        change_count += 1
                        if change_count == 2:
                            index = i + 5  # Apply offset from smoothening
                            arr.append(index)
                            break
                elif data[i] < data[i - 1]:
                    if current_trend is not "down":
                        current_trend = "down"
                        change_count += 1
            # print("Next low index is {}".format(str(index)))
            video.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = video.read()
            frame = cv.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('video/backswing.jpg', frame)

            # # Write this as a video
            # TODO: Add this video creation for the DTL view
            # # Get video properties
            # width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
            # height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
            # fps = int(video.get(cv.CAP_PROP_FPS))
            #
            # fourcc = cv.VideoWriter_fourcc(*'mp4v')
            #
            # out = cv.VideoWriter("swing_stages/{}/backswing.mp4".format(view), fourcc, fps, (width, height))
            #
            # for i in range(prev_index, index):
            #     # Read and write video between these frames
            #     video.set(cv.CAP_PROP_POS_FRAMES, i)
            #     ret, frame = video.read()
            #     out.write(frame)

            state += 1

        if state == 3:
            start_point = np.array([index, data[index]])
            end_point = np.array([np.argmax(data), np.max(data)])

            diff = end_point[0] - start_point[0]
            point = start_point[0] + ((9 / 10) * diff)  # Get the 3/4 marker
            index = int(point) + 1  # get the next frame (works better as its closer to the impact point)
            arr.append(index)

            video.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = video.read()
            frame = cv.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('video/downswing.jpg', frame)
            state += 1

        # Get the frame before and after impact (will interpolate between these), this is going to be the fastest
        # part of the swing.
        current_trend = "up"
        if state == 4:
            for i in range(index, len(data)):  # Allow on offset of 4 for error
                if data[i] < data[i - 1]:
                    current_trend = "down"
                    index = i + 2  # offset
                    arr.append(index)
                    break  # This is the first min

            # Save frame index before impact
            # TODO: Why do we have to subtract a couple of frames here
            video.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = video.read()
            frame = cv.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('video/pre-impact.jpg', frame)
            # Save frame index after impact
            video.set(cv.CAP_PROP_POS_FRAMES, index - state + 1)
            ret, frame = video.read()
            frame = cv.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('video/post-impact.jpg', frame)
            state += 1

            # Get the end of the follow through
            current_trend = "down"
            if state == 5:
                for i in range(index, len(data)):
                    if data[i] > data[i - 1]:
                        current_trend = "up"
                        index = i
                        arr.append(index)
                        break  # This is the first min
                # print("Low index is {}".format(str(index)))

                # Save this frame as an image
                video.set(cv.CAP_PROP_POS_FRAMES, index - state)
                ret, frame = video.read()
                frame = cv.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
                if not isMac:
                    frame = cv.rotate(frame, cv.ROTATE_180)
                cv.imwrite('video/followthrough.jpg', frame)
                state += 1

        # Graph curve and classified frames
        # y = np.arange(len(data))  # [0,1,2,3...N]
        # plt.plot(y, data)
        # # arr.append(len(data) - 1)
        # # temp = [data[x] for x in arr]
        # # plt.scatter(arr, temp, label="Classified frames")
        #
        # plt.ylabel("Movement (pixels)")
        # plt.xlabel("Frame")
        # plt.legend()
        #
        # plt.show()

    def classify(self):
        # Smoothen acceleration curves
        acc = smoothen_curve(self.data)
        acc2 = smoothen_curve(self.data2)

        # Classsify stages
        self.classify_stages(acc, self.video_a, FO)  # pass face on data and save
        self.classify_stages(acc2, self.video_b, DTL)  # pass down the line data and save
