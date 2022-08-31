from sqlalchemy import column
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title('Predicting Fraud in Financial Payment Services')
    

col1, col2, col3 = st.columns(3)

with col1:

    step = st.number_input('Insert the step (1 step is 1 hour of time)', step=1)
    amount = st.number_input('amount of the transaction')
    option = st.selectbox('Select the type of the transaction',('TRANSFER', 'CASH-OUT'))
    if option=="TRANSFER":
        type = 0
    else:
        type = 1
with col2:
    oldBalanceOrig = st.number_input('Initial balance before the transaction')
    newBalanceOrig = st.number_input('Customer\'s balance after the transaction')
with col3:
    oldBalanceDest = st.number_input('Initial recipient balance before the transaction')
    newBalanceDest = st.number_input('Recipient\'s balance after the transaction')

def predict(data):
    host = 'localhost'
    port = '8001'
    url = f'http://{host}:{port}/invocations'
    headers = {
        'Content-Type': 'application/json',
    }
    r = requests.post(url=url, headers=headers, data=data)
    return r

if st.button("Predict"):
    data = {
        "i":0, "step": step,"type":type,"amount":amount,"oldBalanceOrig":oldBalanceOrig,"newBalanceOrig":newBalanceOrig,\
        "oldBalanceDest":oldBalanceDest,"newBalanceDest":newBalanceDest,"errorBalanceOrig":newBalanceOrig + amount - oldBalanceOrig, "errorBalanceDest":oldBalanceDest + amount - newBalanceDest
        }
    test_data = pd.DataFrame(data, index=[0])
    http_data = test_data.to_json(orient='split')
    r = predict(http_data)

    if int(r.text[1]) == 0:
        st.warning("This transaction is fraudulent")
    else:
        st.success("This transaction is not fraudulent")

