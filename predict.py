#using faslk for api endpoint creation

from flask import Flask, request 
import pickle
import sklearn
import numpy as np
import pandas as pd
'''
INFO
Unique values and range of each column
term : 2 [' 36 months' ' 60 months']
grade : 7 ['B' 'A' 'C' 'E' 'D' 'F' 'G']
home_ownership : 6 ['RENT' 'MORTGAGE' 'OWN' 'OTHER' 'ANY' 'NONE']
verification_status : 3 ['Not Verified' 'Source Verified' 'Verified']
purpose : 14 ['vacation' 'debt_consolidation' 'credit_card' 'home_improvement'
 'small_business' 'major_purchase' 'other' 'medical' 'wedding' 'car'
 'moving' 'house' 'educational' 'renewable_energy']
initial_list_status : 2 ['w' 'f']
application_type : 3 ['INDIVIDUAL' 'JOINT' 'DIRECT_PAY']
zip_code : 10 ['22690' '05113' '00813' '11650' '30723' '70466' '29597' '48052' '86630'
 '93700']
'''

'''
Please paste the below json in postman along with the url to see the results
JSON
{
  "loan_amnt": 10000.0,
  "term": " 36 months",
  "int_rate": 11.44,
  "installment": 329.48,
  "grade": "B",
  "home_ownership": "RENT",
  "annual_inc": 117000.0,
  "verification_status": "Not Verified",
  "purpose": "vacation",
  "dti": 26.24,
  "open_acc": 16.0,
  "pub_rec": 0,
  "revol_bal": 36369.0,
  "revol_util": 41.8,
  "total_acc": 25.0,
  "initial_list_status": "w",
  "application_type": "INDIVIDUAL",
  "mort_acc": 0,
  "pub_rec_bankruptcies": 0,
  "zip_code": "22690"
}
'''


app = Flask(__name__)
print(__name__)

with open('log_ress_model.pickle', 'rb') as file:
    model = pickle.load(file)

def pub_rec(number):
    if number == 0.0:
        return 0
    else:
        return 1
    
def mort_acc(number):
    if number == 0.0:
        return 0
    elif number >= 1.0:
        return 1
    else:
        return number
    
    
def pub_rec_bankruptcies(number):
    if number == 0.0:
        return 0
    elif number >= 1.0:
        return 1
    else:
        return number

@app.route('/',methods=['GET'])
def hello():
   return "<p>Hello, World!</p>"

@app.route('/ping', methods=['GET'])
def pinger():
    return {'MESSAGE':'Hello pinger...!!!'}

@app.route('/predict',methods=['POST'])
def predict():

    loanTap_req = request.get_json()
    print(loanTap_req)
    #1.mapping term values
    term_values={' 36 months': 36, ' 60 months':60}
    if loanTap_req['term'] == ' 36 months':
        loanTap_req['term']=36
    else:
        loanTap_req['term']=60

    list_status = {'w': 0, 'f': 1}
    if loanTap_req['initial_list_status'] == 'w':
        loanTap_req['initial_list_status']=0
    else:
        loanTap_req['initial_list_status']=1

    loanTap_req['pub_rec']= pub_rec(loanTap_req['pub_rec'])
    loanTap_req['mort_acc']=mort_acc(loanTap_req['mort_acc'])
    loanTap_req['pub_rec_bankruptcies']=pub_rec_bankruptcies(loanTap_req['pub_rec_bankruptcies'])
    print("loantap data:",loanTap_req)

    #converting json to dataframe
    data= pd.DataFrame([loanTap_req])
    print("dataFrame:", data)

    #2.encoding
    cat_cols_en=['grade', 'home_ownership', 'verification_status', 'purpose',
       'application_type', 'zip_code']
    with open('encoder_object.pickle', 'rb') as file:
        encoder = pickle.load(file)
    print(encoder)
    encoded_data=encoder.transform(data[cat_cols_en])
    print('raw encoded:',encoded_data)
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols_en))
    data_en = pd.concat([data,encoded_df], axis=1)
    data_en.drop(columns=cat_cols_en, inplace=True)
    data_en.dropna(inplace=True)
    print("encoded data:", data_en)

    #3.Scaling
    with open('scaler_object.pickle', 'rb') as file:
        scaler = pickle.load(file)
    X_test = pd.DataFrame(scaler.transform(data_en), columns=list(data_en))
    print("scaled data:",X_test)
    
    #taking only the values from the json dict and converting into 2d array which is taken as the i/p for model prediction
    result = model.predict(X_test)

    #4.Unmapping the mapped result
    if result ==0:
        pred = 'Fully Paid'
    else:
        pred = 'Charged Off'
        
    return {'loan_status':pred}
# run this file using below command on the terminal
# FLASK_APP=predict.py flask run
