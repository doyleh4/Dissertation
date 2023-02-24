import math
import tqdm

import cv2
import numpy as np

from helper_funcs.pose_estimation import PoseEstimation

# pause = [38]
# pause2 = [28]


def find_stop_point(arr):
    max_index = arr.argmax()

    for i in range(max_index, 0, -1):
        if arr[i] < arr[i - 1]:
            return i
    return max_index  # no local min


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
    # res = savgol_filter(res, 30, 3)  # window size 30, polynomial order 6 - good for regular speed
    # res = savgol_filter(res, 180, 3)  # (expected shape, but bumpy) - good for slo-mo but needs more experimentation
    # res = savgol_filter(res, 12, 3)

    kernel_size = 7
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved_10 = np.convolve(res, kernel, mode='same')

    return data_convolved_10


pose = PoseEstimation()

# Load the video file
cap = cv2.VideoCapture('../videos/in_sync/a.MOV')
cap2 = cv2.VideoCapture('../videos/in_sync/b.MOV')

arr = []
arr2 = []
# Check if the video was successfully loaded
if not cap.isOpened() or not cap2.isOpened():
    print('Error loading video file')
    exit()

# Loop through each frame of the video
while cap.isOpened() or cap2.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()

    # Check if there are no more frames
    if not ret or not ret2:
        break

    frame = cv2.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    frame2 = cv2.resize(frame2, (int(frame2.shape[1] / 2.5), int(frame2.shape[0] / 2.5)))
    frame2 = cv2.rotate(frame2, cv2.ROTATE_180)

    try:
        arr.append(pose.get_left_wrist(frame))
        arr2.append(pose.get_left_wrist(frame2))
    except:
        print("Failure to find left wrist in frame")
        arr.append([None, None])
        arr2.append([None, None])

    # if ret:
    #     cv2.imshow("a", frame)
    # if ret2:
    #     cv2.imshow("b", frame2)

    # f_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
    # f_no2 = cap2.get(cv2.CAP_PROP_POS_FRAMES)

    # if f_no in pause or f_no2 in pause2:
    #     print(f_no in pause)
    #     print(f_no2 in pause2)
    #     cv2.waitKey()

    # Display the frame in a window called "video"
    cv2.waitKey(10)

# Release the video capture object and close the window
cap.release()
cap2.release()
cv2.destroyAllWindows()
#
# tempX = [val[0] for val in arr]
# tempY = [val[1] for val in arr]
# # TODO: t - val[1] seems to shove it below the y axis so fix this
# plt.margins(1, 2.8)  # set margins to approximately be the same as opencv window
# curve, = plt.plot(tempX, tempY)
# plt.scatter(tempX[0], tempY[0])
# plt.show()
#
# tempX = [val[0] for val in arr2]
# tempY = [val[1] for val in arr2]
# # TODO: t - val[1] seems to shove it below the y axis so fix this
# plt.margins(1, 2.8)  # set margins to approximately be the same as opencv window
# curve, = plt.plot(tempX, tempY)
# plt.scatter(tempX[0], tempY[0])
# plt.show()

acc = calculate_acceleration(arr)
acc2 = calculate_acceleration(arr2)

a = find_stop_point(acc)
b = find_stop_point(acc2)
#
# fig, axs = plt.subplots(2)
#
# # TODO: t - val[1] seems to shove it below the y axis so fix this
# axs[0].plot(acc)
# axs[0].scatter(a, acc[a])
#
# # TODO: t - val[1] seems to shove it below the y axis so fix this
# axs[1].plot(acc2)
# axs[1].scatter(b, acc2[b])
# plt.show()

print("Processing video")

# Reload the video file
cap = cv2.VideoCapture('../videos/in_sync/a.MOV')
cap2 = cv2.VideoCapture('../videos/in_sync/b.MOV')

# Get 25 frames either side of top of backswing from these indices
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter("synced-a.mp4", fourcc, fps, (width, height))
out2 = cv2.VideoWriter("synced-b.mp4", fourcc, fps, (width, height))

print("First video")
for i in tqdm.tqdm(range(a - 40, a + 40)):
    # Read and write video between these frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    out.write(frame)

print("Second video")
for i in tqdm.tqdm(range(b - 40, b + 40)):
    # Read and write video between these frames
    cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap2.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    out2.write(frame)

out.release()
out2.release()

cap.release()
cap2.release()

print("Allowing time for videos to save")

# time.sleep(30)

# Load the synced videos
cap = cv2.VideoCapture('synced-a.mp4')
cap2 = cv2.VideoCapture('synced-b.mp4')

pose = PoseEstimation()

arr = []

# Loop through each frame of the video
while cap.isOpened() or cap2.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()

    # Check if there are no more frames
    if not ret or not ret2:
        break

    frame = cv2.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    frame2 = cv2.resize(frame2, (int(frame2.shape[1] / 2.5), int(frame2.shape[0] / 2.5)))
    frame2 = cv2.rotate(frame2, cv2.ROTATE_180)

    # try:
    #     arr.append(pose.get_left_wrist(frame))
    #     arr2.append(pose.get_left_wrist(frame2))
    # except:
    #     print("Failure to find left wrist in frame")
    #     arr.append([None, None])
    #     arr2.append([None, None])

    if ret:
        cv2.imshow("a", frame)
    if ret2:
        cv2.imshow("b", frame2)

    # Display the frame in a window called "video"
    cv2.waitKey(20)
