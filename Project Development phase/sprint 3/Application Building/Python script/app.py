import numpy as np 
import pickle 
import joblib 
import matplotlib 
import matplotlib.pyplot as plt 
import time 
import pandas 
import os
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
model = pickle.load(open('C:\\Users\\intec\\rainfall.pkl', 'rb')) 
scale = pickle.load(open('C:\\Users\\intec\\scale.pkl', 'rb'))

@app.route('/')# route to display the home page 
def home():
    return render_template('index.html') #rendering the home page
@app.route('/predict', methods=[ "POST", "GET"])# route to show the predictions in a web UI 
def predict():
    # reading the inputs given by the user 
    input_feature=[x for x in request.form.values() ] 
    features_values=[np.array(input_feature)] 
    names=[['Location','Mintemp','Maxtemp','Rainfall','WindGustSpeed',
    'Windspeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday',
    'WindGustDir', 'WindDir9am', 'WindDir3pm', 'year', 'month', 'day']]
    data = pandas.DataFrame(features_values, columns=names)
    data = scale.fit_transform(data) 
    data = pandas.DataFrame(data,columns = names)
         # predictions using the loaded model file
    prediction = model.predict(data)
    pred_prob = model.predict_proba(data) 
    print(prediction) 
    if prediction == "Yes":
        return render_template("chance.html") 
    else:
        return render_template("nochance.html")
        # showing the prediction results in a UI
if __name__ == "__main__":
    app.run(host='127.0.0.1',debug=False)