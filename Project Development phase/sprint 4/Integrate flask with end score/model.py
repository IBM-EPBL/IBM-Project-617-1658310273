from flask import Flask,render_template,request
import joblib
import numpy as np
import random
import requests
import json
API_KEY="94rOJzRGSmIFTxtASvVhQqEsKGZWE-SPoBkatk88yKJR"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken=token_response.json()["access_token"]
print("mltoken",mltoken)
header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}
app = Flask(__name__)

@app.route('/')# route to display the home page 
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
   l=[]

   l.append(float(request.form['Location']))
   l.append(float(request.form['MinTemp']))
   l.append(float(request.form['Maxtemp']))
   l.append(float(request.form['Rainfall']))
   l.append(float(request.form['WindGustSpeed']))
   l.append(float(request.form['WindSpeed9am']))
   l.append(float(request.form['WindSpeed3pm']))
   l.append(float(request.form['Humidity9am']))
   l.append(float(request.form['Humidity3pm']))
   l.append(float(request.form['Pressure9am']))
   l.append(float(request.form['Pressure3pm']))
   l.append(float(request.form['Temp9am']))
   l.append(float(request.form['Temp3pm']))
   l.append(float(request.form['year']))
   l.append(float(request.form['month']))
   l.append(float(request.form['day']))
   l.append(float(request.form['Rain Today']))
   l.append(float(request.form['WindGustDir']))
   l.append(float(request.form['WindDir9am']))
   l.append(float(request.form['WindDir3pm']))
   print(l)
   payload_scoring = {"input_data": [{"field": [["Location","MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine","WindGustDir","WindGustSpeed","WindDir9am","WindDir3pm","WindSpeed9am","WindSpeed3pm","Humidity9am",]],
                   "values":[l]}]}
   response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/47407097-2468-40a4-9b7b-db84bf05f7ad/predictions?version=2022-11-19', json=payload_scoring,
   headers={'Authorization': 'Bearer ' + mltoken})
   print("Scoring response")
   predictions=response_scoring.json()
   pred=predictions['predictions'][0]['values'][0][0]
   if(pred):
       pred="chances of rain today"
   else:
       pred="no chances of rain today,enjoy your outing"
   return renter_template('index.html', prediction_text = pred)

if __name__ == "__main__":
    app.run(debug=False,port=5000)