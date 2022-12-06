﻿# Predicting-Fraud-in-Financial-Payment-Services
Build and deploy a model that predicts whether a transaction is a fraud or not
This project is a machine learning project that involves building and deploying a model that can predict whether a given transaction is a fraud or not. This project is built with several tools and technologies to accomplish it, including Docker, Scikit learn, MLFlow, and Streamlit.

Docker is used to package the project as a self-contained, portable application. This allows the project to be easily deployed and run on any system that supports Docker, without having to worry about installing and configuring the required dependencies.

Scikit learn is a popular machine learning library for Python. It is used in the project to build and train the fraud detection model. Sklearn provides a wide range of algorithms and tools for building and evaluating machine learning models, making it a powerful and versatile choice for this project.

MLFlow is an open-source platform for managing the machine learning lifecycle. It is used in the project to track and monitor the model training process, and to manage the various versions of the model that are generated during the development process.

Finally, Streamlit is a library for building interactive, web-based applications with Python. It is used in the project to create a user-friendly interface for interacting with the fraud detection model. The interface allows users to input transaction data and view the predicted fraud probability for each transaction.
## Screenshots

![Home Page](/images/1.png "Home Page")

## Setup
- First, build an MLflow docker image with all dependencies
```bash
docker build -t mlflow . 
```
- Run docker compose (MLflow, MINIO, PostgreSQL)
```bash
docker-compose up
```

- Create a bucket in [minio](http://localhost:9001) to store artifacts (name it mlflow)
- Create an experiment in [mlflow](http://localhost:5000) to store artifacts (name it for example FraudPredicition)
- Set required environment variables
```bash
$Env:MLFLOW_TRACKING_URI='http://localhost:5000'  (PowerShell)
export MLFLOW_TRACKING_URI='http://localhost:5000' (Unix-like)
```

- Run some experiments 
```bash
cd ./Project
mlflow run . --experiment-name FraudPredicition -P n_estimators=10 -P max_samples=0.5 -P max_features=0.75 -P max_depth=10
```
- Register your best model in the mlflow registry
- serve the model registred via a RESTAPI
```bash
docker exec -d mlflow mlflow models serve --model-uri models:/<Register-name>/Production -h 0.0.0.0 -p 8001 --no-conda
```

- Launche the web application
```bash
cd ./app
virtualenv venv
./venv/Scripts/activate (Windows)
source ./venv/bin/activate (Linux)
```
```bash
pip install -r requirements.txt
```
```bash
streamlit run main.py
```
## links
- **Mlflow:** http://localhost:5000
- **Minio:** http://localhost:9001
- **Streamlit:** http://localhost:8501/

## Built With

- Docker
- Sklearn
- MLFlow
- Streamlit
- Minio
- Postgres


## Author

**Youssef EL ASERY**

- [Profile](https://github.com/ysfesr "Youssef ELASERY")
- [Linkedin](https://www.linkedin.com/in/youssef-elasery/ "Welcome")
- [Kaggle](https://www.kaggle.com/youssefelasery "Welcome")


## 🤝 Support

Contributions, issues, and feature requests are welcome!

Give a ⭐️ if you like this project!
