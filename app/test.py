import requests
import pandas as pd



url = f'http://localhost:8001/invocations'
headers = {"Content-Type": "application/json; charset=utf-8"}  #{'Content-Type, application/json',}

data = pd.DataFrame({"i":0,"step": 12,"type":2,"amount":122344,"oldBalanceOrig":222,"newBalanceOrig":3333,"oldBalanceDest":33333,"newBalanceDest":4444,"errorBalanceOrig":928, "errorBalanceDest":8872}, index=[0])
data = data.to_json(orient='split')
r = requests.post(url=url, headers=headers, data=data)
print(r.text)












# import requests 

# data = {
#     "step": 12,"type":type,"amount":122344,"oldBalanceOrig":222,"newBalanceOrig":3333,\
#     "oldBalanceDest":33333,"newBalanceDest":4444,"errorBalanceOrig":928, "errorBalanceDest":8872
#     }

# host = 'localhost'
# port = '8001' 
# url = f'http://{host}:{port}/invocations' 
# headers = {'Content-Type': 'application/json',} 
# # test_data is a Pandas dataframe with data for testing the ML model
# http_data = data
# r = requests.post(url=url, headers=headers, data=http_data) 
# print(f'Predictions: {r.text}')