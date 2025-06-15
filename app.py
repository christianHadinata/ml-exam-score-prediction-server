from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Inisialisasi Flask app
app = Flask(__name__)
CORS(app)

# Load model dan scaler dari file
model = joblib.load("best_multiple_linear_model_student.pkl")
scaler = joblib.load("scaler.pkl")

# Daftar fitur yang digunakan oleh model
features = ['study_hours_per_day', 'mental_health_rating',
            'netflix_hours', 'social_media_hours', 'exercise_frequency']


@app.route("/")
def home():
    return "API untuk Prediksi Exam Score"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil input user dari JSON body
        input_data = request.get_json()

        # Validasi input
        if not all(feature in input_data for feature in features):
            return jsonify({
                "error": "Input tidak lengkap. Diperlukan fitur: " + ", ".join(features)
            }), 400

        # Ubah ke DataFrame agar bisa diproses
        input_df = pd.DataFrame([input_data])

        # Normalisasi input dengan scaler yang telah dilatih
        input_scaled = scaler.transform(input_df)

        # Prediksi menggunakan model
        prediction = model.predict(input_scaled)

        return jsonify({
            "predicted_exam_score": max(0.0, min(round(float(prediction[0]), 2), 100.0))
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
