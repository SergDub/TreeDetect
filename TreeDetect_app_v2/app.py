from flask import Flask, request, render_template, send_file
from deepforest import main
from PIL import Image, ImageDraw
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Loading the trained model
model = main.deepforest.load_from_checkpoint("model/trained_DF_model.pth") 

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Getting the uploaded file
        if "file" not in request.files:
            return "No file was uploaded"
        file = request.files["file"]
        if file.filename == "":
            return "File name is empty"
        
        # Saving the uploaded file
        input_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.png")
        file.save(input_path)
        
        # Processing the image
        image = Image.open(input_path)
        predictions = model.predict_image(path = input_path)

        # Drawing bounding boxes
        draw = ImageDraw.Draw(image)
        for index, row in predictions.iterrows():
            draw.rectangle(
                [(row["xmin"], row["ymin"]), (row["xmax"], row["ymax"])],
                outline="red",
                width=3
            )
        # Saving the image with bounding boxes
        result_path = os.path.join(RESULT_FOLDER, f"result_{os.path.basename(input_path)}")
        image.save(result_path)
        
        # Calculating the number of trees
        tree_count = len(predictions)
        print(predictions)

        return render_template("index.html", result_image=result_path, tree_count=tree_count)

    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)