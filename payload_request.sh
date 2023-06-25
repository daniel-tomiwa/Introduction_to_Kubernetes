#!/usr/bin/bash

curl -d '{"PULocationID": 10, "DOLocationID": 50, "trip_distance": 40}' \
     -H "Content-Type: application/json" \
     -X POST http://192.168.253.68:8000/predict