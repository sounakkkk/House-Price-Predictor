import json
import pickle
import joblib
from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the pre-trained scaler and regression model
with open('scaling.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('regmodel.pkl', 'rb') as f:
    regression = pickle.load(f) ## the above two lines of code made sure the indifference version of numpy error is fixed
 
@app.route('/')
def home():
    return "House Price Prediction API"

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print("Received Data:", data)
        
        # Ensure the data is in the correct format (list of 13 values)
        input_data = np.array(list(data.values())).reshape(1, -1)
        
        # Check if the input data has 13 features
        if input_data.shape[1] != 13:
            return jsonify({"error": "Expected 13 features, but received {}".format(input_data.shape[1])}), 400
        
        # Scale the data
        new_data = scaler.transform(input_data)
        
        # Predict using the regression model
        output = regression.predict(new_data)
        print("Prediction:", output[0])
        
        # Convert output to list for JSON serialization
        output_list = output.tolist()
        
        return jsonify(output_list)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)



