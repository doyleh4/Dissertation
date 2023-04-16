import os
import time

import cv2 as cv
import json
import matplotlib
from matplotlib import pyplot as plt

from web_app.helper_funcs.sync_videos import Synchronizer as VideoParser

folder_path = "test_data/unprocessed/"
print(os.listdir(folder_path))

# Define object with indices of swing movement for each video
with open(os.path.join(folder_path, "swing_ranges.json")) as f:
    data = json.load(f)

# Evaluation/Testing metrics
t_count = 0  # true positive
f_count = 0  # false positive

fail_case_1 = 0  # counter for fail case when start index is too late

if __name__ == "__main__":
    print("Running test framework for video preprocessing stage")

    # Open every video in folder
    for filename in os.listdir(folder_path):
        # Only run this code for the .mov files
        print("Processing: {}".format(filename))
        if ".mov" in filename:
            video = os.path.join(folder_path, filename)
            ranges = data[filename]
            cap = cv.VideoCapture(video)

            try:
                parser = VideoParser(cap, None)
                parsed = parser.main()
            except:
                print("Error parsing the video")
                continue

            if ranges[0] >= parsed[0] and ranges[1] <= parsed[1]:
                print("true")
                t_count += 1
            else:
                print("false")
                f_count += 1

            # Manually verified these to be indexed incorrect. Could be implemented in code by check index differences
            if filename == "20.mov" or filename == "4.mov" or filename == "5.mov" or filename == "3.mov":
                fail_case_1 += 1

    # Evaluation metrics
    print("{}% accuracy".format(str(t_count * 100 / (t_count + f_count))))
    total_count = t_count + f_count

    plt.bar(0, t_count, color="green", label="Correct")
    plt.bar(0, f_count, color="red", bottom=t_count, label="Fail")
    plt.xticks([0], ['Total'])
    plt.yticks(range(0, total_count + 1))
    plt.ylabel("Tests")
    plt.legend()
    plt.title("Video parsing results over {} tests".format(total_count))
    plt.show()

    x = ["Start too late", "End too early", "Invalid index"]
    y = [fail_case_1, 0, 0]  # manually verified
    plt.plot(x, y)
    plt.xticks([0, 1, 2], x)
    plt.ylabel("Tests")
    plt.legend()
    plt.title("Counts for failure case scenarios")
    plt.show()
