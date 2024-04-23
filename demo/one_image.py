import os

import cv2
from aaa_image_enhancement.defects_detection_fns import (
    classical_detectors,
)
from aaa_image_enhancement.enhance_image import EnhanceAgentFirst, ImageEnhancer
from aaa_image_enhancement.image_defects_detection import (
    DefectsDetector,
)
from aaa_image_enhancement.image_utils import ImageConversions
from flask import Flask, redirect, render_template, request, send_file

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
upload_folder = os.path.join(app.root_path, UPLOAD_FOLDER)
os.makedirs(upload_folder, exist_ok=True)
app.config["UPLOAD_FOLDER"] = upload_folder

defects_detector = DefectsDetector(classical_detectors)


@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)

        file = request.files["image"]

        if file.filename == "":
            return redirect(request.url)

        img_path = os.path.join(app.config["UPLOAD_FOLDER"], str(file.filename))
        file.save(img_path)

        img = cv2.imread(img_path)

        defects = defects_detector.find_defects(ImageConversions(img))

        image_enhancer = ImageEnhancer(img)
        enhance_agent = EnhanceAgentFirst(img, image_enhancer, defects)

        enhanced_img = enhance_agent.enhance_image()

        enhanced_img_path = os.path.join(
            app.config["UPLOAD_FOLDER"], "enhanced_" + str(file.filename)
        )
        cv2.imwrite(enhanced_img_path, enhanced_img)

        return render_template(
            "index.html",
            defects=defects,
            original_image=file.filename,
            enhanced_image="enhanced_" + str(file.filename),
        )

    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename))


if __name__ == "__main__":
    app.run(debug=True)
