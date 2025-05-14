from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
import traceback

# Load models
preprocessor = joblib.load("preprocessor.pkl")
log_reg = joblib.load("log_reg.pkl")
rand_forest = joblib.load("rand_forest.pkl")
optimizer_lr = joblib.load("optimizer_lr.pkl")
optimizer_rf = joblib.load("optimizer_rf.pkl")
exponentGra = joblib.load("mitigator.pkl")

app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])

        # Preprocess input
        X = preprocessor.transform(df)
        sensitive_feature = df["sex"]
        group = df["sex"].values[0]

        # Base model probabilities
        log_prob = float(log_reg.predict_proba(X)[0][1])
        rf_prob = float(rand_forest.predict_proba(X)[0][1])

        # Raw model predictions
        log_pred = int(log_reg.predict(X)[0])
        rf_pred = int(rand_forest.predict(X)[0])

        # ThresholdOptimizer predictions
        olr_pred = int(optimizer_lr.predict(X, sensitive_features=sensitive_feature)[0])
        orf_pred = int(optimizer_rf.predict(X, sensitive_features=sensitive_feature)[0])

        # Mitigation model (e.g. Exponentiated Gradient)
        exponentGra_pred = int(exponentGra.predict(X)[0])
        exponentGra_prob = (
            float(exponentGra.predict_proba(X)[0][1])
            if hasattr(exponentGra, "predict_proba")
            else None
        )

        # Threshold & confidence for Optimized LR
        thresholds_lr = optimizer_lr.interpolated_thresholder_.interpolation_dict
        group_threshold_lr = thresholds_lr.get(group, 0.5)
        fair_adjusted_confidence_lr = log_prob - group_threshold_lr

        # Threshold & confidence for Optimized RF
        thresholds_rf = optimizer_rf.interpolated_thresholder_.interpolation_dict

        group_threshold_rf = thresholds_rf.get(group, 0.5)
        fair_adjusted_confidence_rf = rf_prob - group_threshold_rf

        return jsonify(
            {
                "logistic_regression": {
                    "prediction": log_pred,
                    "probability": round(log_prob, 4),
                },
                "random_forest": {
                    "prediction": rf_pred,
                    "probability": round(rf_prob, 4),
                },
                "optimized_lr": {
                    "prediction": olr_pred,
                    "probability": round(log_prob, 4),
                    "threshold": round(group_threshold_lr, 4),
                    "adjusted_confidence": round(fair_adjusted_confidence_lr, 4),
                },
                "optimized_rf": {
                    "prediction": orf_pred,
                    "probability": round(rf_prob, 4),
                    "threshold": round(group_threshold_rf, 4),
                    "adjusted_confidence": round(fair_adjusted_confidence_rf, 4),
                },
                "mitigation": {
                    "prediction": exponentGra_pred,
                    "probability": (
                        round(exponentGra_prob, 4)
                        if exponentGra_prob is not None
                        else None
                    ),
                },
            }
        )

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
