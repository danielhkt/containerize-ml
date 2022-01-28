FROM python:3.9

RUN mkdir /home/docker_data
ENV DATA_DIR=/home/docker_data
ENV MODEL_FILE=reg_model.pkl
ENV SCALER_FILE=scaler.gz
ENV DATA_FILE=test.csv

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY scaler.gz ./scaler.gz
COPY reg_model.pkl ./reg_model.pkl
COPY inference.py ./inference.py
