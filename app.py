import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import math
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))



@app.route('/')
def home():
    
    mydata = pd.read_excel('Covid-19.xlsx')
    plt.plot(mydata)


    plt.savefig('static/images/plot.png')
    
    sns.boxplot(mydata['Deaths'],mydata['Test'])
    
    plt.savefig('static/images/kdeplot.png')
    
    sns.boxplot(mydata['Test'],mydata['Infected'])
    
    plt.savefig('static/images/kdeplot2.png')
    
    return render_template('home.html', url='/static/images/plot.png', url1='/static/images/kdeplot.png', url2='/static/images/kdeplot2.png')
    

@app.route('/predict', methods=['POST'])
def predict():
    int_feauters = int(request.form.get('day'))
    
    final_feauters = int_feauters+245
   
    
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