from application import app
from flask import render_template, request, json, jsonify
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import requests
import numpy
import pandas as pd

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/churnclassify",methods =['GET', 'POST'])
def churnclassify():

    #extract form inputs
    SeniorCitizen = request.form.get("SeniorCitizen")
    tenure = request.form.get("tenure")
    gender_Male = request.form.get("gender_Male")
    Dependents_Yes = request.form.get("Dependents_Yes")
    input_data = json.dumps({"SeniorCitizen":SeniorCitizen,"tenure": tenure, "gender_Male":gender_Male, "Dependents_Yes": Dependents_Yes})
    #url for churn classification api
    url = "http://localhost:8082/api"
    #url = "https://cmu95829-churn-predictor-17c35f3572b4.herokuapp.com/api"

    #post data to url
    results =  requests.post(url, input_data)

    #send input values and prediction result to index.html for display 
    return render_template("index.html", SeniorCitizen = SeniorCitizen,tenure = tenure, gender_Male = gender_Male, \
        Dependents_Yes = Dependents_Yes,results = results.content.decode('UTF-8'))