import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import math
import sklearn

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')
    

@app.route('/predict', methods=['POST'])
def predict():
    int_feauters = int(request.form.get('day'))
    
    final_feauters = int_feauters+201
   
    
    prediction = model.predict([[final_feauters]])

    
    prediction = prediction.tolist()
    return render_template('predict.html',prediction=prediction, dayofpred=int_feauters)

@app.route('/about')
def about():
    return render_template('about.html')
 
@app.route('/contact')
def contact():
    return render_template('contact.html')
 
 
if __name__ == '__main__':
    app.run(debug=True)