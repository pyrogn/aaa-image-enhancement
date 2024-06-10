"""App for demonstration of automatic image enhancement."""

import logging
import os
from logging import Logger

import requests
from flask import Flask, redirect, render_template, request, send_file

logger = Logger(__name__, logging.DEBUG)
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
upload_folder = os.path.join(app.root_path, UPLOAD_FOLDER)
os.makedirs(upload_folder, exist_ok=True)
app.config["UPLOAD_FOLDER"] = upload_folder


@app.route("/", methods=["GET", "POST"])
def upload_images():
    if request.method == "POST":
        files = request.files.getlist("images")
        if not files:
            return redirect(request.url)

        enhanced_images = []
        for file in files:
            if file.filename == "":
                continue

            img_path = os.path.join(app.config["UPLOAD_FOLDER"], str(file.filename))
            file.save(img_path)

            with open(img_path, "rb") as img_file:
                img_data = img_file.read()

            enhance_response = requests.post(
                "http://main_app:8000/enhance_image", files={"image": img_data}
            )

            enhanced_img_path = None
            if enhance_response.status_code == 200:
                enhanced_img_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], "enhanced_" + str(file.filename)
                )
                with open(enhanced_img_path, "wb") as f:
                    f.write(enhance_response.content)

            enhanced_images.append(
                {
                    "original": file.filename,
                    "enhanced": "enhanced_" + str(file.filename)
                    if enhanced_img_path
                    else None,
                }
            )

        return render_template("index.html", enhanced_images=enhanced_images)
    logger.info("get")
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
