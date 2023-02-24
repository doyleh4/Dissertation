import math

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

import my_config.config as config

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


def calculate_acceleration(filled):
    res = []
    for i in range(1, len(filled) - 1):
        try:
            dist = math.dist(filled[i - 1], filled[i])
            res.append(dist)
        except:
            print("None value encountered")

    # Smoothen this to make the classification easier
    # TODO: This savgol filer parameters could be changed if the frame rate is differnet (i.e. slo-mo camera), this result
    # doesnt turn out similar in slo-mo videos
    res = savgol_filter(res, 30, 3)  # window size 30, polynomial order 6 - good for regular speed
    # res = savgol_filter(res, 180, 3)  # (expected shape, but bumpy) - good for slo-mo but needs more experimentation
    # res = savgol_filter(res, 12, 3)

    # kernel_size = 7
    # kernel = np.ones(kernel_size) / kernel_size
    # res = np.convolve(res, kernel, mode='same')
    return res


class StageClassifier:
    def __init__(self, processed_data, video_feed, video_feed_b):
        self.data = processed_data
        self.video_a = video_feed
        self.video_b = video_feed_b

        self.video_a.set(cv.CAP_PROP_POS_FRAMES, 0)
        self.video_b.set(cv.CAP_PROP_POS_FRAMES, 0)

    def classify(self):
        acc = calculate_acceleration(self.data)
        # y = np.arange(len(acc))
        # plt.plot(y, acc)
        #
        # # TODO NOTE: This graph is the one that shows the classification curve of the acceleration. Needs to be in report
        # plt.show()

        # NOTE: When we are saving the images here, the imwrite() function extracts the frame from the video
        # hence we need to take this into account when indexing the video after an imwrite(). Hence the index - states
        # when saving frames
        arr = []
        index = 0
        state = 0
        current_trend = "down"
        # Get setup by getting the lowest acceleration point at the start
        if state == 0:
            for i in range(1, len(acc)):
                if acc[i] > acc[i - 1]:
                    current_trend = "up"
                    index = i
                    # arr.append(index)
                    break  # This is the first min
            # print("Low index is {}".format(str(index)))

            # Save this frame as an image
            self.video_a.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = self.video_a.read()
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/face_on/setup.jpg', frame)
            self.video_b.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = self.video_b.read()
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/dtl/setup.jpg', frame)
            state += 1

        # Get takeaway by detecting first significant change in the data values.
        if state == 1:
            threshold = 1.5

            # Compute differnce in values of the array
            temp = acc[index:index+20]  # get a slice of the array starting at index and go 10 frames
            diff = np.abs(np.diff(temp))
            h = np.max(np.where(diff > threshold))  # highest val above thresh

            # Get first instance of this (apply offset)
            index = index + h
            # arr.append(index)

            # Save this frame as an image
            self.video_a.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = self.video_a.read()
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/face_on/takeaway.jpg', frame)
            self.video_b.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = self.video_b.read()
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/dtl/takeaway.jpg', frame)
            state += 1

        # Get the top of the backswing by detecting the next low apex
        # Note: Due to the smoothening of the graph we need to offset this by roughly 5 frames
        # TODO: Fix this so it doesnt have to be offset
        current_trend = "up"
        if state == 2:
            prev_index = index
            change_count = 0

            for i in range(index, len(acc)):
                if acc[i] > acc[i - 1]:
                    if current_trend is not "up":
                        current_trend = "up"
                        change_count += 1
                        if change_count == 2:
                            index = i + 5  # Apply offset from smoothening
                            # arr.append(index)
                            break
                elif acc[i] < acc[i - 1]:
                    if current_trend is not "down":
                        current_trend = "down"
                        change_count += 1
            # print("Next low index is {}".format(str(index)))
            self.video_a.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = self.video_a.read()
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/face_on/backswing.jpg', frame)
            self.video_b.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = self.video_b.read()
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/dtl/backswing.jpg', frame)

            # # Write this as a video
            # TODO: Add this video creation for the DTL view
            # # Get video properties
            # width = int(self.video.get(cv.CAP_PROP_FRAME_WIDTH))
            # height = int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT))
            # fps = int(self.video.get(cv.CAP_PROP_FPS))
            #
            # fourcc = cv.VideoWriter_fourcc(*'mp4v')
            #
            # out = cv.VideoWriter("swing_stages/face_on/backswing.mp4", fourcc, fps, (width, height))
            #
            # for i in range(prev_index, index):
            #     # Read and write video between these frames
            #     self.video.set(cv.CAP_PROP_POS_FRAMES, i)
            #     ret, frame = self.video.read()
            #     if not isMac:
            #         frame = cv.rotate(frame, cv.ROTATE_180)
            #     out.write(frame)

            state += 1

            # TODO: Add a classfier for downswing!
        if state == 3:
            start_point = np.array([index, acc[index]])
            end_point = np.array([np.argmax(acc), np.max(acc)])
            # index = np.mean([start_point, end_point], axis=0)[0]
            # points = np.linspace(start_point, end_point, 2, end_point=False)
            # index = points[int(3 / 4 * 2)]  # Get point 3/4 the way

            diff = end_point[0] - start_point[0]
            point = start_point[0] + ((4/5) * diff)  # Get the 3/4 marker
            index = int(point)-1 # get index of this point
            # arr.append(index)
            # print(index)
            # y = np.arange(len(acc))
            # plt.plot(y, acc)
            # plt.scatter(start_point[0], start_point[1])
            # plt.scatter(end_point[0], end_point[1])
            # plt.scatter([index], acc[index])
            # plt.show()

            self.video_a.set(cv.CAP_PROP_POS_FRAMES, index)
            ret, frame = self.video_a.read()
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/face_on/downswing.jpg', frame)
            self.video_b.set(cv.CAP_PROP_POS_FRAMES, index)
            ret, frame = self.video_b.read()
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/dtl/downswing.jpg', frame)
            state += 1

        # Get the frame before and after impact (will interpolate between these), this is going to be the fastest
        # part of the swing.
        current_trend = "up"
        if state == 4:
            # y = np.arange(len(acc))
            # plt.plot(y, acc)
            # plt.scatter(index, acc[index])
            # plt.show()
            for i in range(index, len(acc)): # Allow on offset of 4 for error
                if acc[i] < acc[i - 1]:
                    current_trend = "down"
                    index = i + 2 # offset
                    arr.append(index)
                    break  # This is the first min
            # print("High index is {}".format(str(index)))

            # Save frame index before impact
            # TODO: Why do we have to subtract a couple of frames here
            self.video_a.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = self.video_a.read()
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/face_on/pre-impact.jpg', frame)
            self.video_b.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = self.video_b.read()
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/dtl/pre-impact.jpg', frame)
            # Save frame index after impact
            self.video_a.set(cv.CAP_PROP_POS_FRAMES, index - state + 1)  # +1 to get the frame after impact
            ret, frame = self.video_a.read()
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/face_on/post-impact.jpg', frame)
            self.video_b.set(cv.CAP_PROP_POS_FRAMES, index - state + 1)
            ret, frame = self.video_b.read()
            if not isMac:
                frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/dtl/post-impact.jpg', frame)
            state += 1

            # Get the end of the follow through
            current_trend = "down"
            if state == 5:
                for i in range(index, len(acc)):
                    if acc[i] > acc[i - 1]:
                        current_trend = "up"
                        index = i
                        # arr.append(index)
                        break  # This is the first min
                # print("Low index is {}".format(str(index)))

                # Save this frame as an image
                self.video_a.set(cv.CAP_PROP_POS_FRAMES, index)
                ret, frame = self.video_a.read()
                if not isMac:
                    frame = cv.rotate(frame, cv.ROTATE_180)
                cv.imwrite('swing_stages/face_on/followthrough.jpg', frame)
                self.video_b.set(cv.CAP_PROP_POS_FRAMES, index)
                ret, frame = self.video_b.read()
                if not isMac:
                    frame = cv.rotate(frame, cv.ROTATE_180)
                cv.imwrite('swing_stages/dtl/followthrough.jpg', frame)
                state += 1

        # y = np.arange(len(acc))
        # plt.plot(y, acc)
        #
        # temp = [acc[x] for x in arr]
        # plt.scatter(arr, temp)
        #
        # TODO NOTE: This graph is the one that shows the classification curve of the acceleration. Needs to be in report
        # plt.show()