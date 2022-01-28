# Containerize your ML model using Docker and perform batch inference

This code demonstrates how to containerize your ML model and perform inference using Docker by means of a simple example.

Steps:
1. Create & prepare a synthetic dataset and train a regression model.
    
    -> python train.py

2. Build the Docker image adding the data transformer, trained model, and inference code (see Dockerfile).
    
    -> docker build -t docker-ml-model -f Dockerfile .

3. Perform inference by running the container.
    
    -> docker run -v /Users/<placeholder>/docker_data/:/home/docker_data/ docker-ml-model python inference.py

The inference data is stored locally and we serve it by mounting a volume (-v) to the container, i.e. mapping the local directory to the container.
Consequentially, we store the prediction outputs in the same local directory.

This is just a simple example, but in practice we would prefer to decouple and containerize each step separately, i.e. a separate container for data preparation, model training, and inference.


Prerequisites:
- Docker Desktop
- Install dependencies in train.py
