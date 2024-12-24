from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import gradio as gr

LABELS = os.listdir("../HandGesture/images")
model1 = load_model("./models/cnn_basic.keras")
model2 = load_model("./models/vgg16.hdf5")
model3 = load_model("./models/handgest_transfer4.hdf5")
model4 = load_model("./models/MobileNetV2.keras")
model5 = load_model("./models/CNN_with_dropout.keras")

def predict_image(image, select_model):
    models = {
        "CNN from scratch": model1,
        "VGG16": model2,
        "DenseNet201": model3,
        "MobileNetV2": model4,
        "CNN_with_dropout": model5
    }
    if select_model == "CNN from scratch" or select_model == "CNN_with_dropout":
        images_label = {i: label for i, label in enumerate(LABELS)}
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (195, 195))
        image = image.astype(np.float32) / 255.
        image = np.expand_dims(image, axis=0)
        prediction = models[select_model].predict(image)
        predicted_class = np.argmax(prediction, axis=1)
        if np.max(prediction) > 0.8:
            return images_label[predicted_class[0]]
        else:
            return "Not Related"
    elif select_model == "VGG16" or select_model == "DenseNet201":
        labels = ['scissor', 'thumbs', 'paper', 'rock', 'rock_on', 'fingers_crossed', 'call_me', 'up', 'okay', 'peace']
        image = cv2.resize(image, (60, 60))
        image = image.astype(np.float32) / 255.
        image = np.expand_dims(image, axis=0)
        prediction = models[select_model].predict(image)
        predicted_class = np.argmax(prediction, axis=1)
        if np.max(prediction) > 0.7:
            return labels[predicted_class[0]]
        else:
            return "Not Related"
    elif select_model == "MobileNetV2":
        labels = ['peace', 'rock_on', 'fingers_crossed', 'up', 'thumbs', 'rock', 'scissor', 'paper', 'okay', 'call_me']
        image = cv2.resize(image, (195, 195))
        image = image.astype(np.float32) / 255.
        image = np.expand_dims(image, axis=0)
        prediction = models[select_model].predict(image)
        predicted_class = np.argmax(prediction, axis=1)
        if np.max(prediction) > 0.8:
            return labels[predicted_class[0]]
        else:
            return "Not Related"


iface = gr.Interface(fn=predict_image, inputs=[gr.Image(), gr.Dropdown(choices=["CNN from scratch", "VGG16", "DenseNet201", "MobileNetV2", "CNN_with_dropout"], label="Select Model")], outputs="label", title="Hand Gesture Detection", example_labels=LABELS)
iface.launch()