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
    gender = request.form.get("gender")
    SeniorCitizen = request.form.get("SeniorCitizen")
    Partner = request.form.get("Partner")
    Dependents = request.form.get("Dependents")
    tenure = request.form.get("tenure")
    PhoneService = request.form.get("PhoneService")
    MultipleLines = request.form.get("MultipleLines")
    InternetService= request.form.get("InternetService")
    OnlineSecurity = request.form.get("OnlineSecurity")
    OnlineBackup = request.form.get("OnlineBackup")
    DeviceProtection = request.form.get("DeviceProtection")
    TechSupport = request.form.get("TechSupport")
    StreamingTV= request.form.get("StreamingTV")
    StreamingMovies = request.form.get("StreamingMovies")
    Contract = request.form.get("Contract")
    PaperlessBilling = request.form.get("PaperlessBilling")
    PaymentMethod = request.form.get("PaymentMethod")
    MonthlyCharges = request.form.get("MonthlyCharges")
    TotalCharges = request.form.get("TotalCharges")


    #extract data from json
    input_data = json.dumps({"gender":gender, "SeniorCitizen":SeniorCitizen, "Partner": Partner, "Dependents": Dependents, "tenure": tenure, \
        "PhoneService": PhoneService, "MultipleLines": MultipleLines, "InternetService": InternetService, \
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup, \
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport, "StreamingTV": StreamingTV, \
        "StreamingMovies": StreamingMovies, "Contract": Contract, "PaperlessBilling": PaperlessBilling, \
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges})

    #url for churn classification api
    #url = "http://localhost:5000/api"
    url = "https://churn-predict-265aa0298bfd.herokuapp.com/api"

 
    #post data to url
    results =  requests.post(url, input_data)

    #send input values and prediction result to index.html for display
    return render_template("index.html", gender = gender, SeniorCitizen = SeniorCitizen, Partner = Partner, Dependents = Dependents, tenure = tenure, \
        PhoneService = PhoneService, MultipleLines = MultipleLines, InternetService = InternetService, OnlineSecurity = OnlineSecurity, \
        OnlineBackup = OnlineBackup, DeviceProtection = DeviceProtection, TechSupport = TechSupport, \
        StreamingTV = StreamingTV, StreamingMovies = StreamingMovies, Contract = Contract, \
        PaperlessBilling = PaperlessBilling, PaymentMethod = PaymentMethod, MonthlyCharges = MonthlyCharges, TotalCharges = TotalCharges, \
        results = results.content.decode('UTF-8'))