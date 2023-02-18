import math

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

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
    res = savgol_filter(res, 30, 3)  # window size 30, polynomial order 6 (expected shape, but bumpy)
    # res = savgol_filter(res, 30, 6)  # window size 30, polynomial order 6 (expected shape, but bumpy)
    return res


class StageClassifier:
    def __init__(self, processed_data, video_feed):
        self.data = processed_data
        self.video = video_feed

        self.video.set(cv.CAP_PROP_POS_FRAMES, 0)

    def classify(self):
        acc = calculate_acceleration(self.data)
        y = np.arange(len(acc))
        plt.plot(y, acc)
        plt.show()

        ## Using the slope to classify the satge isnt great, better to use the apex of curve
        # slopes = []
        # for i in range(1, len(acc)):
        #     slopes.append((acc[i] - acc[i - 1]) / (y[i] - y[i - 1]))
        #
        # y = np.arange(len(slopes))
        # plt.plot(y, slopes)
        # plt.show()

        # NOTE: When we are saving the images here, the imwrite() function extracts the frame from the video
        # hence we need to take this into account when indexing the video after an imwrite(). Hence the index - states
        # when saving frames

        index = 0
        state = 0
        current_trend = "down"
        # Get setup by getting the lowest acceleration point at the start
        if state == 0:
            for i in range(1, len(acc)):
                if acc[i] > acc[i - 1]:
                    current_trend = "up"
                    index = i
                    break  # This is the first min
            # print("Low index is {}".format(str(index)))

            # Save this frame as an image
            self.video.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = self.video.read()
            # TODO: FInd out why it doesnt rotate on mac but does on windows
            frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/setup.jpg', frame)
            state += 1

        # Get takeaway by detecting first significant change in the data values.
        if state == 1:
            threshold = 1

            # Compute differnce in values of the array
            diff = np.abs(np.diff(acc))
            indices = np.where(diff > threshold)[0]  # [0] will retrun as an array

            # Get first instance of this
            index = indices[0]

            # Save this frame as an image
            # TODO: FInd out why it doesnt rotate on mac but does on windows
            self.video.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = self.video.read()
            # TODO: FInd out why it doesnt rotate on mac but does on windows
            frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/takeaway.jpg', frame)
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
                            break
                elif acc[i] < acc[i - 1]:
                    if current_trend is not "down":
                        current_trend = "down"
                        change_count += 1
            # print("Next low index is {}".format(str(index)))
            self.video.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = self.video.read()
            # TODO: FInd out why it doesnt rotate on mac but does on windows
            frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/backswing.jpg', frame)

            # # Write this as a video
            # # Get video properties
            # width = int(self.video.get(cv.CAP_PROP_FRAME_WIDTH))
            # height = int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT))
            # fps = int(self.video.get(cv.CAP_PROP_FPS))
            #
            # fourcc = cv.VideoWriter_fourcc(*'mp4v')
            #
            # out = cv.VideoWriter("swing_stages/backswing.mp4", fourcc, fps, (width, height))
            #
            # for i in range(prev_index, index):
            #     # Read and write video between these frames
            #     self.video.set(cv.CAP_PROP_POS_FRAMES, i)
            #     ret, frame = self.video.read()
            #     # TODO: FInd out why it doesnt rotate on mac but does on windows
            #     frame = cv.rotate(frame, cv.ROTATE_180)
            #     out.write(frame)

            state += 1

            # Get the frame before and after impact (will interpolate between these), this is going to be the fastest
            # part of the swing.
            current_trend = "up"
        if state == 3:
            for i in range(index, len(acc)):
                if acc[i] < acc[i - 1]:
                    current_trend = "down"
                    index = i
                    break  # This is the first min
            # print("High index is {}".format(str(index)))

            # Save frame index before impact
            # TODO: Why do we have to subtract a couple of frames here
            self.video.set(cv.CAP_PROP_POS_FRAMES, index - state)
            ret, frame = self.video.read()
            # TODO: FInd out why it doesnt rotate on mac but does on windows
            frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/pre-impact.jpg', frame)
            # Save frame index after impact
            self.video.set(cv.CAP_PROP_POS_FRAMES, index - state + 1)  # +1 to get the frame after impact
            ret, frame = self.video.read()
            # TODO: FInd out why it doesnt rotate on mac but does on windows
            frame = cv.rotate(frame, cv.ROTATE_180)
            cv.imwrite('swing_stages/post-impact.jpg', frame)

            state += 1

            # Get the end of the follow through
            current_trend = "down"
            if state == 4:
                for i in range(index, len(acc)):
                    if acc[i] > acc[i - 1]:
                        current_trend = "up"
                        index = i
                        break  # This is the first min
                # print("Low index is {}".format(str(index)))

                # Save this frame as an image
                self.video.set(cv.CAP_PROP_POS_FRAMES, index - state)
                ret, frame = self.video.read()
                # TODO: FInd out why it doesnt rotate on mac but does on windows
                frame = cv.rotate(frame, cv.ROTATE_180)
                cv.imwrite('swing_stages/followthrough.jpg', frame)
                state += 1
