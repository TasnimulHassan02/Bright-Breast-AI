from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf 
from PIL import Image
import io
import base64
from io import BytesIO
from matplotlib import pyplot as plt


app = Flask(__name__)
model = tf.keras.models.load_model('multi_task_U-Net_model.h5') 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file.stream).resize((256, 256))
    original_img = img.convert('RGB')  # Displayed as is
    gray_img = img.convert('L')

    # Preprocess
    img_array = np.expand_dims(np.array(gray_img) / 255.0, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    segmentation_output, classification_output = model.predict(img_array)
    predicted_class = np.argmax(classification_output)
    class_names = {0: "Normal", 1: "Benign", 2: "Malignant"}
    class_name = class_names[predicted_class]

    # Create mask image
    mask = segmentation_output[0, :, :, 0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).convert("L").resize(original_img.size)

    # Convert original image to base64
    original_buffer = BytesIO()
    original_img.save(original_buffer, format="PNG")
    original_base64 = base64.b64encode(original_buffer.getvalue()).decode("utf-8")

    # Convert mask image to base64
    mask_buffer = BytesIO()
    mask_img.save(mask_buffer, format="PNG")
    mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")

    return render_template(
        'index.html',
        prediction=f"Predicted class: {class_name}",
        original_image=original_base64,
        mask_image=mask_base64
    )


if __name__ == '__main__':
    app.run(debug=True)
