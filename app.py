from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS, cross_origin
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the saved models and preprocessing objects
turnover_model = joblib.load('turnover_model.pkl')
promotion_model = joblib.load('promotion_model.pkl')
scaler = joblib.load('scaler.pkl')
one_hot_encoder = joblib.load('one_hot_encoder.pkl')  # Load the saved OneHotEncoder

@app.route('/turnover', methods=['POST'])
@cross_origin()
def predict_turnover():
    try:
        # Get input data from the request
        data = request.json

        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])

        # Preprocess categorical features using OneHotEncoder
        categorical_features = ['Department', 'EducationLevel', 'JobTitle', 'DaysSinceStart']
        categorical_data = input_data[categorical_features]

        print("1")

        # Transform categorical data using OneHotEncoder
        encoded_categorical_data = one_hot_encoder.transform(categorical_data)

        # Combine encoded categorical data with numerical features
        numerical_features = ['Age', 'YearsOfService', 'NumberOfLeavesTaken', 'AttendancePercentage', 'PerformanceRating', 'LeavePercentage', 'DaysSinceStart']
        numerical_data = input_data[numerical_features]
        
        # Scale numerical features
        scaled_numerical_data = scaler.transform(numerical_data)

        # Combine all features
        processed_data = np.hstack([encoded_categorical_data, scaled_numerical_data])

        # Make prediction
        prediction = turnover_model.predict(processed_data)
        print(prediction)

        return jsonify({"prediction": int(prediction[0])}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/promotion', methods=['POST'])
@cross_origin()
def predict_promotion():
    try:
        # Get input data from the request
        data = request.json

        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])

        # Preprocess categorical features using OneHotEncoder
        categorical_features = ['Department', 'EducationLevel', 'JobTitle', 'DaysSinceStart']
        categorical_data = input_data[categorical_features]

        # Transform categorical data using OneHotEncoder
        encoded_categorical_data = one_hot_encoder.transform(categorical_data)

        # Combine encoded categorical data with numerical features
        numerical_features = ['Age', 'YearsOfService', 'NumberOfLeavesTaken', 'AttendancePercentage', 'PerformanceRating', 'LeavePercentage', 'DaysSinceStart']
        numerical_data = input_data[numerical_features]

        # Scale numerical features
        scaled_numerical_data = scaler.transform(numerical_data)

        # Combine all features
        processed_data = np.hstack([encoded_categorical_data, scaled_numerical_data])

        # Make prediction
        prediction = promotion_model.predict(processed_data)


        return jsonify({"prediction": int(prediction[0])}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)