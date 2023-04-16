"""
This script is used to extract frames from the video and save them as a JPG, top be used in testing

"""
import cv2 as cv

video_path = "../videos/sample/sample6.mov"
test_dir_path = "../testing/ground_truth/frames"

# Frames we want to save

frame_indexs = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70]
# Open video once not every iteration
cap = cv.VideoCapture(video_path)
if __name__ == "__main__":
    # Open video once not every iteration
    cap = cv.VideoCapture(video_path)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    for i in frame_indexs:
        if i < total_frames:
            # Set cap to specified frame
            cap.set(cv.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            # save this frame
            if ret:
                # Rotate frame in the same manner as in main script to retain dimensions
                # TODO: update this once I find out why it only needs to be done on windows
                frame = cv.resize(frame, (int(frame.shape[1] / 2.5), int(frame.shape[0] / 2.5)))
                frame = cv.rotate(frame, cv.ROTATE_180)

                # print("{}/{}".format(test_dir_path, i))
                cv.imwrite("{}/{}.jpg".format(test_dir_path, i), frame)
                # cv.imshow("Frame", frame)
                # cv.waitKey(0)
        else:
            print("Index: {} is out of total frame count: {}".format(i, total_frames))

    # Release the video capture object
    cap.release()

    # Close all windows
    cv.destroyAllWindows()
