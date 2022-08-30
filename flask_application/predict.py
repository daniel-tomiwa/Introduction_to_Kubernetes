import pickle
import os
from flask import Flask, request, jsonify

with open('lin_reg.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

def preprocess(ride):
    features = {}
    features['PU_DO'] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
    pred = model.predict(X)
    return pred[0]

app = Flask("duration-prediction")

@app.route("/")
def home():
    html = f"<h2> Welcome to the home page for duratio prediction </h2>"
    return html.format(format)

@app.route("/predict", methods=['POST'])
def predict_endpoint():
    json_payload = request.get_json()

    features = preprocess(json_payload)
    pred = predict(features)

    prediction_json = {
        "duration": pred
    }
    return jsonify(prediction_json)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)