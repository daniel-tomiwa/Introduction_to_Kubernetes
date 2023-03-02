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
    username = os.environ.get('DB_USERNAME', "postgres")
    password = os.environ.get('DB_PASSWORD', "password")
    html = f"""<h1 align='centre'> Welcome to the home page for taxi ride duration prediction </h1>
               <h3> The username: {username} </h3> 
               <h3> The password: {password} </h3>
            """
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