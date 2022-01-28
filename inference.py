#!/usr/bin/python

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error

# set environment variables
MODEL_FILE = os.environ["MODEL_FILE"]
SCALER_FILE = os.environ["SCALER_FILE"]
DATA_FILE = os.environ["DATA_FILE"]
DATA_DIR = os.environ["DATA_DIR"]
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)


def inference():
    # load docker_data and transform docker_data
    data = pd.read_csv(DATA_PATH)
    scaler = load(SCALER_FILE)
    data_scaled = scaler.transform(data)

    X, y = data_scaled[:, :-1], data_scaled[:, -1]

    # load model, make and evaluate predictions
    reg_model = load(MODEL_FILE)
    preds = reg_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print("RMSE: %f" % rmse)

    # save predictions
    output_path = os.path.join(DATA_DIR, 'output.csv')
    pd.DataFrame(preds).to_csv(output_path)


if __name__ == '__main__':
    inference()
