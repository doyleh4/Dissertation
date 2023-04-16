# Helper imports
import os
import time

# Required imports
import cv2 as cv
import numpy as np
from flask import Flask, render_template, Response, send_file, request, jsonify

# Custom class imports
from helper_funcs.analyse_swing import DTLAnalyser, FOAnalyser
from helper_funcs.classify_stage import StageClassifier
from helper_funcs.data_record import FODataRecord as FOData, DTLDataRecord as DTLData
from helper_funcs.feedback_system import NegativeFeedback as Feedback
from helper_funcs.graphing import GraphHelper as Graph
from helper_funcs.sync_videos import Synchronizer as VideoParser
from main import face_on_view, down_the_line

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 500  # 500 MB limit


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/upload_video", methods=["POST"])
def upload_video():
    # This function saves the uploaded video to repo in server
    tags = ["FO-video", "DTL-video"]
    for tag in tags:
        try:
            file = request.files.get(tag)
            if file:
                try:
                    # Download the file in the request file tag
                    filename = file.filename
                    file.save(filename)

                    # Rename the file to be consistent between uploads
                    try:
                        new_name = "{}.MOV".format(tag)
                        if os.path.exists(new_name):
                            os.remove(new_name)
                            time.sleep(1)
                        os.rename(filename, new_name)
                    except Exception as e:
                        print(e)
                    print("{} - File downloaded successfully".format(tag))
                except Exception as e:
                    print(e)
        except:
            continue
    return jsonify({'success': True})


# Route to serve the video file
@app.route('/video/<path:filename>')
def video(filename):
    video_path = "video/{}".format(filename)
    return send_file(video_path, mimetype='video/mp4')


@app.route('/FO_analysis')
def fo_analysis():
    cap = cv.VideoCapture("./FO-video.MOV")

    parser = VideoParser(cap, None)
    parsed = parser.main()

    # Init vars
    data = FOData()
    draw = Graph()

    # Process video
    data = face_on_view("./video/temp_parsed.mp4")

    # Calculate acceleration
    acc = draw.get_acceleration(data)

    # Classify stages
    print("Video is being classified for its stages")
    view = cv.VideoCapture('./video/temp_parsed.mp4')
    classifier = StageClassifier(acc, view)
    classifier.single_classify()

    # Analyse the images
    analyser = FOAnalyser()
    res = analyser.analyse()

    # Get feedback for results
    feedback = Feedback(res)
    feedback.process()

    # Convert the Point element to be compatible with JS and HTML
    for i in res:
        # Process differently depending on items in point
        temp = np.array(i["Points"])
        temp = np.ndim(temp)
        if temp == 2:
            converted = [item.tolist() for item in i["Points"]]
        elif temp == 3:
            converted = [[item.tolist() for item in temp] for temp in i["Points"]]
        # for item in i["Points"]:
        #     t = item.tolist()
        i["Points"] = converted

    return render_template("FO_analysis.html", results=res)


@app.route("/DTL_analysis")
def dtl_analysis():
    cap = cv.VideoCapture("./DTL-video.MOV")

    parser = VideoParser(cap, None)
    parsed = parser.main()

    data = DTLData()
    draw = Graph()

    data = down_the_line("./video/temp_parsed.mp4")

    # Calculate acceleration
    acc = draw.get_acceleration(data)

    # Classify stages
    print("Video is being classified for its stages")
    view = cv.VideoCapture('./video/temp_parsed.mp4')
    classifier = StageClassifier(acc, view)
    classifier.single_classify()

    # Analyse the images
    analyser = DTLAnalyser()
    res = analyser.analyse()

    feedback = Feedback(res)
    feedback.process()

    # Convert the Point element to be compatible with JS and HTML
    for i in res:
        # Process differently depending on items in point
        temp = np.array(i["Points"])
        temp = np.ndim(temp)
        if temp == 2:
            converted = [item.tolist() for item in i["Points"]]
        elif temp == 3:
            converted = [[item.tolist() for item in temp] for temp in i["Points"]]
        # for item in i["Points"]:
        #     t = item.tolist()
        i["Points"] = converted

    return render_template("DTL_analysis.html", results=res)


@app.route('/play_video')
def play_video():
    return Response(face_on_view(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
