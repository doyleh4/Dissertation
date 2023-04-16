import cv2

cap = cv2.VideoCapture('../testing/test_data/unprocessed/20.mov')
# cap = cv2.VideoCapture('../video/temp_parsed.mp4')
paused = False
video_ended = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()

    if not ret:
        video_ended = True
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Draw the frame number on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    thickness = 2
    frame = cv2.putText(frame, f'Frame: {current_frame}', (20, 50), font,
                        font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.imshow('Video', frame)

    key = cv2.waitKey(25) & 0xFF

    if key == ord(' '):  # Pause/resume playback
        paused = not paused
    elif paused and key == ord('a'):  # Step backward 1 frame
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        new_frame = max(0, current_frame - 3)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        ret, frame = cap.read()
        if not ret:
            video_ended = True
            break
        cv2.imshow('Video', frame)
    elif key == 27:  # Quit on ESC key
        break

if not video_ended:
    cap.release()

cv2.destroyAllWindows()