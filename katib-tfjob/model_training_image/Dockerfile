FROM python:3.10.4-slim-buster

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt /app

COPY ./random_job.py /app

RUN pip install -r requirements.txt
