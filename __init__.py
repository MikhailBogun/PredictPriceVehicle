from flask import render_template, url_for, redirect, jsonify, request, Flask
from pycaret.regression import *
import numpy as np
import pandas as pd
import os
p = os.path.abspath('')
app = Flask(__name__)
model = load_model(p+'/predictFunctions/vehicleCatboostModell')
vehicle_data = pd.read_csv(p+'/predictFunctions/vehicle_price(2020).csv')
cols =[i for i in vehicle_data.columns if i!='price$']

@app.route('/')
def home():    
    return render_template('home.html',pred='quality')

##return Response("It works!"), 200

@app.route('/predict', methods=['POST'])
def predict():
    int_features= [x for x in request.form.values()]
    final= np.array(int_features)
    #final = ['ford','transit',0,'other',0,'diesel',120, "manual"]
    data_unseen= pd.DataFrame([final], columns=cols)
    prediction=predict_model(model, data=data_unseen)
    prediction=prediction.Label
    
    return render_template('home.html',pred='{}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)