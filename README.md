# Predicting-Fraud-in-Financial-Payment-Services
Build and deploy a model that predicts whether a transaction is a fraud or not
This project is a machine learning project that involves building and deploying a model that can predict whether a given transaction is a fraud or not. This project is built with several tools and technologies to accomplish it, including Docker, Scikit learn, MLFlow, and Streamlit.

Docker is used to package the project as a self-contained, portable application. This allows the project to be easily deployed and run on any system that supports Docker, without having to worry about installing and configuring the required dependencies.

Scikit learn is a popular machine learning library for Python. It is used in the project to build and train the fraud detection model. Sklearn provides a wide range of algorithms and tools for building and evaluating machine learning models, making it a powerful and versatile choice for this project.

MLFlow is an open-source platform for managing the machine learning lifecycle. It is used in the project to track and monitor the model training process, and to manage the various versions of the model that are generated during the development process.

Finally, Streamlit is a library for building interactive, web-based applications with Python. It is used in the project to create a user-friendly interface for interacting with the fraud detection model. The interface allows users to input transaction data and view the predicted fraud probability for each transaction.
## Screenshots

![Home Page](/images/1.png "Home Page")

## Setup
- Create an MLflow Docker image containing all necessary dependencies.
```bash
docker build -t mlflow . 
```
- Launch MLflow, MINIO, and PostgreSQL using Docker Compose.
```bash
docker-compose up
```

- Create a bucket named "mlflow" in [minio](http://localhost:9001) to store artifacts.
- Create an MLFlow experiment called "FraudPrediction" and store any artifacts created within it.
- Set up the necessary environment variables
```bash
$Env:MLFLOW_TRACKING_URI='http://localhost:5000'  (PowerShell)
export MLFLOW_TRACKING_URI='http://localhost:5000' (Unix-like)
```

- Using MLFlow, run some experiments to determine the best parameters for your project. 
```bash
cd ./Project
mlflow run . --experiment-name FraudPredicition -P n_estimators=10 -P max_samples=0.5 -P max_features=0.75 -P max_depth=10
```
- Register the best model in the MLflow Registry
- Serve the model registered via a REST API
```bash
docker exec -d mlflow mlflow models serve --model-uri models:/<Register-name>/Production -h 0.0.0.0 -p 8001 --no-conda
```

- Launch the web application
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
