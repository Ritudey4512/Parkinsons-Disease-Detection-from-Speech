from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "models/parkinsons_knn_pipeline.pkl"
DATA_PATH = "data.csv"


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")

except Exception as e:
    model = None
    print("Error loading model:", e)


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE"
]


# ============================================================
# LOAD DATASET
# ============================================================

try:
    dataset = pd.read_csv(DATA_PATH)

    # Make sure all expected columns exist
    missing_columns = [
        column for column in FEATURES + ["status"]
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in data.csv: {missing_columns}"
        )

    print(
        f"Dataset loaded successfully: "
        f"{len(dataset)} samples"
    )

except Exception as e:
    dataset = None
    print("Error loading dataset:", e)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# RANDOM SAMPLE ROUTE
# ============================================================

@app.route("/sample", methods=["GET"])
def get_random_sample():

    if dataset is None:
        return jsonify({
            "error": "Dataset could not be loaded."
        }), 500

    try:

        # Select one random row from the complete dataset
        random_row = dataset.sample(
            n=1
        ).iloc[0]

        # Create response containing only the 22 features
        sample_data = {}

        for feature in FEATURES:
            sample_data[feature] = float(
                random_row[feature]
            )

        # Dataset row number
        sample_number = int(
            random_row.name
        ) + 1

        return jsonify({
            "sample": sample_data,
            "sample_number": sample_number,
            "total_samples": len(dataset)
        })

    except Exception as e:

        return jsonify({
            "error": f"Could not load sample: {str(e)}"
        }), 500


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "Model could not be loaded."
        }), 500

    try:

        # Get JSON data from website
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No input data received."
            }), 400


        # ----------------------------------------------------
        # Check all 22 features
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in FEATURES
            if feature not in data
        ]

        if missing_features:
            return jsonify({
                "error": (
                    "Missing features: "
                    + ", ".join(missing_features)
                )
            }), 400


        # ----------------------------------------------------
        # Convert input to numbers
        # ----------------------------------------------------

        values = []

        for feature in FEATURES:

            try:

                value = float(
                    data[feature]
                )

            except (ValueError, TypeError):

                return jsonify({
                    "error":
                        f"Invalid value for {feature}."
                }), 400

            if not np.isfinite(value):

                return jsonify({
                    "error":
                        f"Invalid numeric value for {feature}."
                }), 400

            values.append(value)


        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [values],
            columns=FEATURES
        )


        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = int(
            model.predict(input_data)[0]
        )


        # ----------------------------------------------------
        # Get probability
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            input_data
        )[0]


        # Find probability corresponding to
        # predicted class

        class_probabilities = dict(
            zip(
                model.classes_,
                probabilities
            )
        )

        predicted_probability = (
            class_probabilities[prediction] * 100
        )


        # Round probability
        predicted_probability = round(
            predicted_probability,
            2
        )


        # ----------------------------------------------------
        # Result messages
        # ----------------------------------------------------

        if prediction == 1:

            result = (
                "Parkinson's-related "
                "pattern detected"
            )

            message = (
                "The model found a voice-feature "
                "pattern associated with the "
                "Parkinson's class in its training data."
            )

        else:

            result = (
                "Parkinson's-related "
                "pattern not detected"
            )

            message = (
                "The model did not find a voice-feature "
                "pattern associated with the Parkinson's "
                "class."
            )


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return jsonify({

            "prediction": prediction,

            "result": result,

            "probability":
                predicted_probability,

            "message": message

        })


    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "error":
                f"Prediction failed: {str(e)}"
        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )