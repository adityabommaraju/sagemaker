FROM python:3.9-slim-buster

WORKDIR /opt/ml/code

ADD train.py .
ADD requirements.txt .

RUN pip install -r requirements.txt
