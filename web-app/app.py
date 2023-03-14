from flask import Flask, render_template, Response, send_file

from main import face_on_view

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


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
    # width, height = face_on_view()
    width, height = (480, 860)
    return render_template("FO_analysis.html")


@app.route("/DTL_analysis")
def dtl_analysis():
    # width, height = down_the_line()
    width, height = (480, 860)
    return render_template("DTL_analysis.html")


@app.route('/play_video')
def play_video():
    return Response(face_on_view(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
