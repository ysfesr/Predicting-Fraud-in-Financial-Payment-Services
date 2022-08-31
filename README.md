# Predicting-Fraud-in-Financial-Payment-Services
Build and deploy a model that predicts whether a transaction is a fraud or not

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