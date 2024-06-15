from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#load data
dataLoc ="../Sample_Data/Raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(dataLoc,sep = ',')

# Drop any columns not needed for prediction
df2 = df.drop(['customerID'], axis=1)

# Drop missing values if any
df2 = df2.dropna()

# Converting Total Charges to a numerical data type.
df2.TotalCharges = pd.to_numeric(df2.TotalCharges, errors='coerce')

#Convertin the churn results into a binary numeric variable
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

# Convert categorical columns to numerical using one-hot encoding
df3 = pd.get_dummies(df2, drop_first=True)

# Split the data into features and target variable
X = df3.drop('Churn', axis=1)
Y = df3['Churn']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate the model
Y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(Y_test, Y_pred)}')
print(classification_report(Y_test, Y_pred))

# Save the model and scaler to disk
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

#create flask instance
app = Flask(__name__)
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
# Load the trained model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')


#==========================
#create api

@app.route('/api', methods=['GET', 'POST'])

def predict():
    #get data from request
    data = request.get_json(force=True)
    data_point = np.array([data["gender"], data["SeniorcITIZEN"], data["Partner"], data["Dependents"], \
                data["tenure"], data["PhoneService"], data["MultipleLines"], data["InternetService"], \
                data["OnlineSecurity"], data["OnlineBackup"],data["tenure"], data["DeviceProtection"], \
                data["TechSupport"], data["StreamingTV"],data["StreamingMovies"], data["Contract"], \
                data["PaperlessBilling"], data["PaymentMethod"],data["MonthlyCharges"],data["TotalCharges"]])
    
    data_point.TotalCharges = pd.to_numeric(data_point.TotalCharges, errors='coerce')
    data_final = pd.get_dummies(data_point, drop_first=True)

    #make predicon using model
    prediction = model.predict(data_final)
    return Response(json.dumps(prediction[0]))
