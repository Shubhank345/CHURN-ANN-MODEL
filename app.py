import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from tensorflow.keras.models import load_model
#LOADING MODEL
model=load_model('model.h5')
#LOADING PICKLE FILES of encoders
with open('one_hot.pkl','rb')as file:
    onehot_encoder_geo=pickle.load(file)
with open('label_encoder_gender.pkl','rb')as file:
  label_encoder_gender=pickle.load(file)
with open('scaler.pkl','rb')as file:
    scaler=pickle.load(file)
    #STREAMLIT APP
    st.title('CUSTOMER CHURN PREDICTION')
    # User input
geography = st.selectbox("🌍 Select Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("👤 Select Gender", label_encoder_gender.classes_)
age=st.slider('Age',18,92)
estimate_salary=st.number_input('Estimate Salary')
credit_score=st.number_input("Credit Score")
tenure=st.slider('Tenure',0,15)
balance=st.number_input('Balance')
product=st.slider('Number of Products',1,4)
card=st.selectbox('Has Credit Card',[0,1])
active_member=st.selectbox('Active Member',[0,1])
#INPUT DATA
input_data =pd.DataFrame ({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [product],
    "HasCrCard": [card],
    "IsActiveMember": [active_member],
    "EstimatedSalary": [estimate_salary],
    "Geography": [geography]
})
#ONE HOT ENCODING
geo_encoder=onehot_encoder_geo.transform([[geography]])
geo_encoder_df=pd.DataFrame(geo_encoder,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoder_df],axis=1)
# Drop raw Geography before scaling
if "Geography" in input_data.columns:
    input_data = input_data.drop(columns=["Geography"])

# Scale features
input_data_scaled = scaler.transform(input_data)
prediction=model.predict(input_data_scaled)
prediction_probe=prediction[0][0]
st.write(f"CHURN PROBABILITY    :{prediction_probe}")
if prediction_probe>0.5:
   st.write("Customer will churn")
else:
   st.write("Customer will not churn")




