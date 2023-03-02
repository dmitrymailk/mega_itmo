# import main Flask class and request object
from flask import Flask, request
from flask_cors import CORS

from PIL import Image
from io import BytesIO
import base64

# create the Flask app
app = Flask(__name__)
CORS(app)


@app.route("/analyse-mood/", methods=["POST"])
def query_example():
    if request.method == "POST":
        content = request.json
        image = str(content["image"])
        print(image)
        image = image.replace("data:image/jpeg;base64,", "")
        im = Image.open(BytesIO(base64.b64decode(image)))
        im.save("test.jpg")
        return image
    return f"Был получен {request.method} запрос."


if __name__ == "__main__":
    app.run(debug=True, port=5000)
