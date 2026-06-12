# routes/ml_routes.py — ML Prediction Endpoints

from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required, current_user
from db import supabase
from datetime import date
import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model once
ml_bp = Blueprint('ml', __name__)
model = load_model("arthricare/models/image_model.h5")

# Class labels
classes = ["normal", "osteoarthritis", "rheumatoid"]

# Image prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    # 🔥 ADD THIS LOGIC
    if confidence < 60:   # you can adjust (50–70)
        return "Not Identified", confidence

    return classes[predicted_index], confidence

@ml_bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        input_type = request.form.get('input_type', 'symptoms')

        # =========================
        # SYMPTOMS (TABULAR MODEL)
        # =========================
        if input_type == 'symptoms':
            features = [
                float(request.form.get('pain_score', 0)),
                float(request.form.get('stiffness_mins', 0)),
                float(request.form.get('swelling', 0)),
                float(request.form.get('fatigue', 0)),
                float(request.form.get('age', 0)),
                float(request.form.get('crp', 0)),
                float(request.form.get('esr', 0)),
                float(request.form.get('uric_acid', 0)),
                float(request.form.get('rf_factor', 0)),
            ]

            from ml.hybrid_model import predict_arthritis
            result = predict_arthritis(features)

        # =========================
        # XRAY IMAGE MODEL
        # =========================
        elif input_type == 'xray':
            file = request.files.get('xray_file')

            if file:
                # Ensure upload folder exists
                upload_folder = "arthricare/static/uploads"
                os.makedirs(upload_folder, exist_ok=True)

                filepath = os.path.join(upload_folder, file.filename)
                file.save(filepath)

                predicted_class, confidence = predict_image(filepath)

                result = {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'risk_level': 'moderate',
                    'model_used': 'CNN Image Model',
                    'gradcam_path': None
                }
            else:
                result = {
                    'predicted_class': 'Unknown',
                    'confidence': 0,
                    'risk_level': 'unknown',
                    'model_used': 'None',
                    'gradcam_path': None
                }

        # =========================
        # SAVE TO DATABASE
        # =========================
        supabase.table('ml_predictions').insert({
            'user_id': current_user.id,
            'prediction_date': str(date.today()),
            'input_type': input_type,
            'predicted_class': result['predicted_class'],
            'confidence_score': result['confidence'],
            'risk_level': result.get('risk_level', 'moderate'),
            'model_used': result.get('model_used', 'Hybrid/CNN'),
            'gradcam_path': result.get('gradcam_path')
        }).execute()

        return render_template('ml_prediction.html', result=result, submitted=True)

    return render_template('ml_prediction.html', result=None, submitted=False)


# =========================
# API ENDPOINT
# =========================
@ml_bp.route('/predict/api', methods=['POST'])
@login_required
def predict_api():
    data = request.get_json()
    features = data.get('features', [])

    from ml.hybrid_model import predict_arthritis
    result = predict_arthritis(features)

    return jsonify(result)