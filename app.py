from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import tensorflow as tf
import numpy as np
import os
import json 
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# Create upload folder if it doesn't exist
os.makedirs("static/tests", exist_ok=True)

# Class labels
verbose_name = {
    0: 'Abnormal_Heartbeat',
    1: 'Myocardial_Infarction',
    2: 'Normal_person',
    3: 'History_of_Myocardial_Infarction'
}

# ---------------- Load Models ----------------
model1 = load_model('ecg_cnn.h5')             # Model 1 = CNN
model2 = load_model('vgg16_ecg_model.h5')     # Model 2 = VGG16
model3 = load_model('vgg19_ecg_model.h5')     # Model 3 = VGG19
model4 = load_model('ecg_model_mnv1new.h5')   # Model 4 = MobileNet

# ---------------- Prediction Functions ----------------
def predict_label(img_path, cropped_path, model):
    """Prediction function for Model4 (MobileNet)"""
    top, bottom = 287, 1513
    left, right = 72, 2150

    img = Image.open(img_path)
    img = img.crop((left, top, right, bottom))
    img.save(cropped_path)
    img = img.resize((224, 224))

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]
    return verbose_name[pred_class]

def predict_label_model2(img_path, cropped_path):
    """Prediction function for Model1 (CNN)"""
    top, bottom = 287, 1513
    left, right = 72, 2150

    img = Image.open(img_path)
    img = img.crop((left, top, right, bottom))
    img.save(cropped_path)
    img = img.resize((224, 224))

    img = np.array(img, dtype=np.float32)
    img_array = img / 255.0
    img_array = np.expand_dims(img, axis=0)

    pred = model1.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]
    return verbose_name[pred_class]

def predict_ecg4(img_path, model):
    """Prediction function for Model2 (VGG16) & Model3 (VGG19)"""
    image_size = (224, 224)
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    return verbose_name[pred_class]

# ---------------- Routes ----------------
@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "1234":
            session["logged_in"] = True
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/index", methods=['GET', 'POST'])
def index():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html")

# -------- Prediction Routes --------
@app.route("/submit_model1", methods=['POST'])
def get_output_model1():
    """Prediction with Model1 = CNN"""
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    img = request.files['my_image']
    img_path = "static/tests/" + img.filename
    cropped_path = "static/tests/cropped_model1_" + img.filename
    img.save(img_path)
    predict_result = predict_label_model2(img_path, cropped_path)  # CNN
    return render_template("prediction.html", prediction=predict_result, img_path=cropped_path)

@app.route("/submit_model2", methods=['POST'])
def get_output_model2():
    """Prediction with Model2 = VGG16"""
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    img = request.files['my_image']
    img_path = "static/tests/" + img.filename
    cropped_path = "static/tests/cropped_model2_" + img.filename
    img.save(img_path)

    # Crop
    top, bottom = 287, 1513
    left, right = 72, 2150
    pil_img = Image.open(img_path)
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img.save(cropped_path)

    predict_result = predict_ecg4(cropped_path, model2)  # VGG16
    return render_template("prediction.html", prediction=predict_result, img_path=cropped_path)

@app.route("/submit_model3", methods=['POST'])
def get_output_model3():
    """Prediction with Model3 = VGG19"""
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    img = request.files['my_image']
    img_path = "static/tests/" + img.filename
    cropped_path = "static/tests/cropped_model3_" + img.filename
    img.save(img_path)

    # Crop
    top, bottom = 287, 1513
    left, right = 72, 2150
    pil_img = Image.open(img_path)
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img.save(cropped_path)

    predict_result = predict_ecg4(cropped_path, model3)  # VGG19
    return render_template("prediction.html", prediction=predict_result, img_path=cropped_path)

@app.route("/submit_model4", methods=['POST'])
def get_output_model4():
    """Prediction with Model4 = MobileNet"""
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    img = request.files['my_image']
    img_path = "static/tests/" + img.filename
    cropped_path = "static/tests/cropped_model4_" + img.filename
    img.save(img_path)
    predict_result = predict_label(img_path, cropped_path, model4)  # MobileNet
    return render_template("prediction.html", prediction=predict_result, img_path=cropped_path)

# -------- Performance Routes --------
@app.route("/performance1")
def performance1():
    metrics_path = os.path.join("static", "cnn_model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics = {k: float(metrics.get(k, 0)) for k in ["accuracy", "precision", "recall", "f1_score"]}
    return render_template(
        "performance.html",
        metrics=metrics,
        loss_graph="/static/cnn_loss_accuracy_plot.png",
        accuracy_graph="/static/cnn_accuracy_plot.png",
        confusion_matrix_graph="/static/cnn_confusion_matrix.png"
    )

@app.route("/performance2")
def performance2():
    metrics_path = os.path.join("static", "vgg16_model_metricsvgg16.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics = {k: float(metrics.get(k, 0)) for k in ["accuracy", "precision", "recall", "f1_score"]}
    return render_template(
        "performance.html",
        metrics=metrics,
        loss_graph="/static/vgg16_loss_accuracy_plot.png",
        accuracy_graph="/static/vgg16_accuracy_plot.png",
        confusion_matrix_graph="/static/vgg16_confusion_matrix.png"
    )

@app.route("/performance3")
def performance3():
    metrics_path = os.path.join("static", "vgg19_model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics = {k: float(metrics.get(k, 0)) for k in ["accuracy", "precision", "recall", "f1_score"]}
    return render_template(
        "performance.html",
        metrics=metrics,
        loss_graph="/static/vgg19_loss_accuracy_plot.png",
        accuracy_graph="/static/vgg19_accuracy_plot.png",
        confusion_matrix_graph="/static/vgg19_confusion_matrix.png"
    )

@app.route("/performance4")
def performance4():
    metrics_path = os.path.join("static", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics = {k: float(metrics.get(k, 0)) for k in ["accuracy", "precision", "recall", "f1_score"]}
    return render_template(
        "performance.html",
        metrics=metrics,
        loss_graph="/static/mobilenet_loss_accuracy_plot.png",
        accuracy_graph="/static/mobilenet_accuracy_plot.png",
        confusion_matrix_graph="/static/mobilenet_confusion_matrix.png"
    )

# ------------------- MAIN -------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
