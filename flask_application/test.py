import requests

ride = {
    "PULocationID": 10,	
    "DOLocationID": 50,
    "trip_distance": 40
}

# features_preprocessed = predict.preprocess(ride)
# prediction = predict.predict(features_preprocessed)

url = "http://localhost:8000/predict"
response = requests.post(url, json=ride)

print(response.json())
