#!/usr/bin/bash

curl -d '{"PULocationID": 10, "DOLocationID": 50, "trip_distance": 40}' \
     -H "Content-Type: application/json" \
     -X POST http://107.21.128.117:30939/predict