from flask import Flask, request, send_file

app = Flask(__name__)

UPLOAD_DIR = "/var/uploads/"


@app.route("/download")
def download_file():
    """Serve a previously-uploaded file by name."""
    filename = request.args.get("file", "")
    path = UPLOAD_DIR + filename
    return send_file(path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
