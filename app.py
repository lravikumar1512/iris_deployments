from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask application
app = Flask(__name__)
import joblib

# Define the filename for the exported model
model_filename = 'logistic_regression_model.joblib'

# Save the trained model to a file
joblib.dump(log_reg_model, model_filename)

print(f"Trained Logistic Regression model exported successfully to '{model_filename}'.")
# Load the pre-trained model
try:
    model = joblib.load("logistic_regression_model.joblib")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON body
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        # Validate input
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in JSON"}), 400

        # Convert features to numpy array
        features = np.array(data["features"], dtype=float).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Convert numpy type to native Python type
        return jsonify({
            "prediction": prediction[0].item()
        })

    except ValueError as ve:
        return jsonify({"error": f"Invalid input data: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
