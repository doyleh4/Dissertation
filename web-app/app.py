import os

from flask import Flask, render_template, Response, send_file, request, jsonify

from main import face_on_view

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 200  # 200 MB limit


@app.route('/')
def index():
    # TODO: drag and drop box that will send video to server on upload this will involve updating the new file paths
    #  in the source code to the new uploaded file and send the correct files to the static folder
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
                    print(os.getcwd())
                    file.save(file.filename)
                    print("Downloaded to server")
                except Exception as e:
                    print("ERRRRRRORORRRRR")
                    print(e)
        except:
            continue
    return jsonify({'success': True})


# Route to serve the video file
@app.route('/video/<path:filename>')
def video(filename):
    video_path = "video/{}".format(filename)
    return send_file(video_path, mimetype='video/mp4')
    # return Response(
    #     send_file("./video/{}".format(filename), as_attachment=False, conditional=False),
    #     mimetype='video/mp4'
    # )
    # return send_from_directory('./video/', filename)


@app.route('/FO_analysis')
def fo_analysis():
    # TODO: Synchronise video
    # width, height = face_on_view()
    width, height = (480, 860)
    return render_template("FO_analysis.html")


@app.route("/DTL_analysis")
def dtl_analysis():
    # TODO: Synchronise video
    # width, height = down_the_line()
    width, height = (480, 860)
    return render_template("DTL_analysis.html")


@app.route('/play_video')
def play_video():
    return Response(face_on_view(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
