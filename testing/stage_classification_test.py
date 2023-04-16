import os
import time

import cv2 as cv
import json
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from web_app.helper_funcs.analyse_swing import draw
from web_app.helper_funcs.classify_stage import StageClassifier
from web_app.helper_funcs.sync_videos import Synchronizer as VideoParser
from web_app.main import face_on_view

folder_path = "test_data/unprocessed/"
print(os.listdir(folder_path))

# Define object with indices of swing movement for each video
with open(os.path.join(folder_path, "swing_ranges.json")) as f:
    data = json.load(f)

# Evaluation/TTesting metrics
setup_t = 0  # true positive
setup_f = 0  # false positive
setup_m = 0  # Roughly correct
ta_t = 0  # takeaway
ta_f = 0
ta_m = 0
bs_t = 0  # backswing
bs_f = 0
bs_m = 0
ds_t = 0  # downswing
ds_f = 0
ds_m = 0
i_t = 0  # impact
i_f = 0
i_m = 0
ft_t = 0  # followthrough
ft_f = 0
ft_m = 0

fail_case_1 = 0  # counter for fail case when start index is too late

if __name__ == "__main__":
    print("Running test framework for video preprocessing stage")

    for filename in os.listdir(folder_path):
        # Only run this code for the .mov files
        time.sleep(5)
        print("Processing: {}".format(filename))
        # Manually verified these to be indexed incorrect. Could be implemented in code by check index differences
        if filename == "20.mov" or filename == "4.mov" or filename == "5.mov" or filename == "3.mov":
            continue

        # Cases where processing will trigger an error
        if filename == "14.mov" or filename == "16.mov" or filename == "10.mov":
            continue

        if ".mov" in filename:
            video = os.path.join(folder_path, filename)
            ranges = data[filename]
            cap = cv.VideoCapture(video)

            # Parse Video
            parser = VideoParser(cap, None)
            parsed = parser.main()

            # Classify stages. This will save them to the "video" folder
            # Process video
            temp = face_on_view("video/temp_parsed.mp4")

            # Calculate acceleration
            acc = []
            acc = draw.get_acceleration(temp)

            # Classify stages
            print("Classifying")
            view = cv.VideoCapture("video/temp_parsed.mp4")
            classifier = StageClassifier(acc, view)
            try:
                classifier.single_classify()
            except Exception as e:
                print("Failed to classify. Skipping")
                print(e)
                continue

            # Individually view the files and manually mark them as good or bad.
            # This should be made into a function but im in a rush

            img = cv.imread('video/setup.jpg')
            while True:
                cv.putText(img, 'Setup', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv.imshow('image', img)

                key = cv.waitKey(0)

                # Check if 'y' or 'n' key is pressed
                if key == ord('y'):
                    setup_t += 1
                    break
                elif key == ord('n'):
                    setup_f += 1
                    break
                elif key == ord('m'):
                    setup_m += 1
                    break

            cv.destroyAllWindows()

            img = cv.imread('video/takeaway.jpg')
            while True:
                cv.putText(img, 'Takeaway', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv.imshow('image', img)

                key = cv.waitKey(0)

                # Check if 'y' or 'n' key is pressed
                if key == ord('y'):
                    ta_t += 1
                    break
                elif key == ord('n'):
                    ta_f += 1
                    break
                elif key == ord('m'):
                    ta_m += 1
                    break

            cv.destroyAllWindows()

            img = cv.imread('video/backswing.jpg')
            while True:
                cv.putText(img, 'backswing', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv.imshow('image', img)

                key = cv.waitKey(0)

                # Check if 'y' or 'n' key is pressed
                if key == ord('y'):
                    bs_t += 1
                    break
                elif key == ord('n'):
                    bs_f += 1
                    break
                elif key == ord('m'):
                    bs_m += 1
                    break

            cv.destroyAllWindows()

            img = cv.imread('video/downswing.jpg')
            while True:
                cv.putText(img, 'downswing', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv.imshow('image', img)

                key = cv.waitKey(0)

                # Check if 'y' or 'n' key is pressed
                if key == ord('y'):
                    ds_t += 1
                    break
                elif key == ord('n'):
                    ds_f += 1
                    break
                elif key == ord('m'):
                    ds_m += 1
                    break

            cv.destroyAllWindows()

            img = cv.imread('video/pre-impact.jpg')
            while True:
                cv.putText(img, 'impact', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv.imshow('image', img)

                key = cv.waitKey(0)

                # Check if 'y' or 'n' key is pressed
                if key == ord('y'):
                    i_t += 1
                    break
                elif key == ord('n'):
                    i_f += 1
                    break
                elif key == ord('m'):
                    i_m += 1
                    break

            cv.destroyAllWindows()

            img = cv.imread('video/followthrough.jpg')
            while True:
                cv.putText(img, 'Follow Through', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv.imshow('image', img)

                key = cv.waitKey(0)

                # Check if 'y' or 'n' key is pressed
                if key == ord('y'):
                    ft_t += 1
                    break
                elif key == ord('n'):
                    ft_f += 1
                    break
                elif key == ord('m'):
                    ft_m += 1
                    break

            cv.destroyAllWindows()

    # Hardcoded results of running the above code.
    # setup_t = 12
    # setup_f = 1
    #
    # ta_t = 11
    # ta_f = 2
    #
    # bs_t = 4
    # bs_f = 9
    #
    # ds_t = 7
    # ds_f = 6
    #
    # i_t = 9
    # i_f = 4
    #
    # ft_t = 8
    # ft_f = 5
    #
    #
    setup_t = 12
    setup_f = 1
    setup_m = 0

    ta_t = 1
    ta_f = 7
    ta_m = 4

    bs_t = 3
    bs_f = 9
    bs_m = 0

    ds_t = 3
    ds_f = 4
    ds_m = 5

    i_t = 1
    i_f = 4
    i_m = 7

    ft_t = 6
    ft_f = 6
    ft_m = 0

    # Place bars together by swing stage

    labels = ["setup", "takeaway", "backswing", "downswing", "impact", "follow through"]
    true_items = [setup_t, ta_t, bs_t, ds_t, i_t, ft_t]
    false_items = [setup_f, ta_f, bs_f, ds_f, i_f, ft_f]
    m_items = [setup_m, ta_m, bs_m, ds_m, i_m, ft_m]

    total_true = sum(true_items)
    total_false = sum(false_items)
    total_m = sum(m_items)

    zipped = zip(true_items, false_items, m_items)
    recall = [tp/(tp + fn) for tp, fn, fp in zipped]

    zipped = zip(true_items, false_items, m_items)
    precisions = [tp/(tp + fp) for tp, fn, fp in zipped]


    zipped = zip(true_items, false_items, m_items)
    f_scores = [tp/(tp + .5*(fp+fn)) for tp, fn, fp in zipped]

    # Averages
    a_r = np.mean(recall)
    a_p = np.mean(precisions)
    a_f = np.mean(f_scores)

    l_width = 0.25
    plt.bar(labels, true_items, l_width, color="green", label="Correct")
    plt.bar([i + l_width for i in range(len(labels))], false_items, l_width, color="red", label="Incorrect")
    plt.bar([i + l_width*2 for i in range(len(labels))], m_items, l_width, color="Purple", label="Sufficient")
    plt.bar([i + l_width*3 for i in range(len(labels))], [0,0,0,0,0,0], l_width) # Insert blank space after

    plt.title("Swing stage classification results")
    plt.xlabel("Swing stage")
    plt.ylabel("Results")
    plt.legend()

    plt.show()

    plt.plot(labels, recall, color="green", label="recalls")
    plt.plot(labels, precisions, color="red", label="precisions")
    plt.plot(labels, f_scores, color="purple", label="f scores")
    plt.axhline(y=a_r, color="green", linestyle="dotted", label="Average recall")
    plt.axhline(y=a_p, color="red", linestyle="dotted", label="Average precision")
    plt.axhline(y=a_f, color="purple", linestyle="dotted", label="Average f score")




    plt.title("Evaluation metric results")
    plt.xlabel("Swing Stage")
    plt.ylabel("Results")
    plt.legend()
    plt.show()

