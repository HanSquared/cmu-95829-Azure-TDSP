from application import app
from flask import render_template, request, json, jsonify
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import requests
import numpy
import pandas as pd

#decorator to access the app
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

#decorator to access the service
@app.route("/churnclassify", methods=['GET', 'POST'])
def churnclassify():

    #extract form inputs
    tenure = request.form.get("tenure")
    contract = request.form.get("contract")
    gender = request.form.get("gender")
    partner = request.form.get("partner")
    dependents = request.form.get("dependents")
    monthly-charges = request.form.get("monthly-charges")
    total-charges = request.form.get("total-charges")
    phone-service = request.form.get("phone-service")
    multiple-lines = request.form.get("multiple-lines")
internet-service = request.form.get("internet-service")
online-security = request.form.get("online-security")
online-backup = request.form.get("online-backup")
device-protection = request.form.get("device-protection")
tech-support = request.form.get("tech-support")
streamingTV = request.form.get("streamingTV")
PaperlessBilling = request.form.get("PaperlessBilling")
PaymentMethod = request.form.get("PaymentMethod")
SeniorCitizen = request.form.get("SeniorCitizen")

   #convert data to json
    input_data = json.dumps({"tenure": tenure, "contract": contract, "gender": gender, "partner": partner, "dependents": dependents, " monthly-charges ": monthly-charges, " total-charges ": total-charges, "phone-service": phone-service, "multiple-lines ": multiple-lines, "internet-service ": internet-service, "online-security ": online-security, "online-backup ": online-backup, "device-protection ": device-protection, "tech-support": tech-support, “streaming-tv": streaming-tv, “StreamingMovies": StreamingMovies, “PaperlessBilling ": PaperlessBilling, “PaymentMethod”: PaymentMethod, “SeniorCitizen”: SeniorCitizen})

    #url for churn predictor model
    #url = "http://localhost:5000/api"
    url = "https://churn-model-app.herokuapp.com/api"
  
    #post data to url
    results =  requests.post(url, input_data)

    #send input values and prediction result to index.html for display
    return render_template("index.html",  results=results.content.decode('UTF-8'))
  
