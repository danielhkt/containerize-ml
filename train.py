#!/usr/bin/python3

import os
import pandas as pd
from joblib import dump
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

docker_data_dir = './docker_data'


def create_dataset():
    print("Creating synthetic dataset.")
    # create synthetic time series dataset with 4 features
    X, y = make_regression(n_samples=600, n_features=4, n_informative=3)

    df = pd.DataFrame(X)
    df.columns = ['feat1', 'feat2', 'feat3', 'feat4']
    df['target'] = y

    # split into train/test docker_data
    train = df[:500]
    test = df[500:]

    # save docker_data
    train.to_csv('train.csv', index=False)
    test_path = os.path.join(docker_data_dir, 'test.csv')
    test.to_csv(test_path, index=False)

    return None


def train():
    print("Training XGBoost Regressor.")
    # load training docker_data
    training = "./train.csv"
    data = pd.read_csv(training)

    # rescale docker_data and store scaler
    scaler = MinMaxScaler()
    scaler = scaler.fit(data)
    data_scaled = scaler.transform(data)
    print("Storing docker_data transformer.")
    dump(scaler, 'scaler.gz')

    X, y = data_scaled[:, :-1], data_scaled[:, -1]

    # train model
    reg_model = GradientBoostingRegressor()
    reg_model.fit(X, y)
    
    # serialize model
    print("Serializing model.")
    dump(reg_model, 'reg_model.pkl')

    return None


if __name__ == '__main__':
    #create_dataset() # uncomment if need to create dataset
    train()
