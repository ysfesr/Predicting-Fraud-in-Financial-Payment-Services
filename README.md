# Predicting-Fraud-in-Financial-Payment-Services
Build and deploy a model that predicts whether a transaction is a fraud or not

## Screenshots

![Home Page](/screenshots/1.png "Home Page")

![](/screenshots/2.png)

![](/screenshots/3.png)

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

- Run some experiments (we use in our script the SVM algorithm, so we need to specify the kernel and the c value used by the algorithm)
```bash
docker exec -d mlflow python /mlflow/train.py --kernel linear --C 10 
```
- serve the one of the models created via a RESTAPI
```bash
docker exec -d mlflow mlflow models serve -m s3://mlflow/<id>/<run-id>/artifacts/<experminets-name> -h 0.0.0.0 -p 8001 --no-conda
```

- At the end install Streamlit in your machine and launche the application 
```bash
pip install streamlit
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